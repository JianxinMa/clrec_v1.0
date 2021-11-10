# coding=utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base_modules import add_accuracy_metrics
from base_modules import get_dnn_variable
from base_modules import get_unit_emb_initializer
from base_modules import weighted_softmax
from common_utils import add_or_fail_if_exist
from common_utils import fully_connected
from common_utils import get_shape


def select_topk_vectors_by_scores(multi_vec_emb,  # [B,V,D]
                                  vec_scores,  # [B,V]
                                  topk):
    batch_size, _, emb_dim = get_shape(multi_vec_emb)
    topk = tf.to_int32(topk)
    _, col_indices = tf.nn.top_k(vec_scores, k=topk)  # [B,V] -> [B,K]
    col_indices = tf.to_int64(col_indices)
    row_indices = tf.tile(tf.expand_dims(
        tf.range(0, tf.to_int64(batch_size), dtype=tf.int64), -1), [1, topk])
    indices = tf.stack([tf.reshape(row_indices, [-1]),
                        tf.reshape(col_indices, [-1])], 1)  # [B*K, 2]
    seq_topk_emb = tf.gather_nd(multi_vec_emb, indices)  # [B*K,D]
    seq_topk_emb = tf.reshape(seq_topk_emb, [batch_size, topk, emb_dim])
    return seq_topk_emb


# This may select duplicate heads if there are spill-over effects.
def select_topk_vectors_by_backtrack(multi_vec_emb,  # [B,V,D]
                                     itm_emb,  # [B,T,D]
                                     topk):
    # [B,V,D] x [B,T,D] -> [B,V,T], assuming both inputs are l2-normalized
    vec_itm_scores = tf.matmul(multi_vec_emb, itm_emb, transpose_b=True)
    prob_v = tf.reduce_max(vec_itm_scores, axis=-1)  # [B,V,T] -> [B,V]
    seq_topk_emb = select_topk_vectors_by_scores(multi_vec_emb, prob_v, topk)
    return seq_topk_emb


def multipath_fully_connected(x, path_out_dim=None, use_bias=True, scope='multi_fc', reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        batch_size, num_path, emb_dim = get_shape(x)
        if path_out_dim is None:
            path_out_dim = emb_dim
        w = get_dnn_variable('weight', shape=[1, num_path, emb_dim, path_out_dim],
                             partitioner=partitioner)
        w = tf.tile(w, [batch_size, 1, 1, 1])  # [B,V,D,D']
        # [B,V,1,D]x[B,V,D,D']=[B,V,1,D']=[B,V,D']
        y = tf.squeeze(tf.matmul(tf.expand_dims(x, 2), w), [2])  # [B,V,D']
        if use_bias:
            b = get_dnn_variable('bias', shape=[1, num_path, path_out_dim],
                                 partitioner=partitioner,
                                 initializer=tf.zeros_initializer())
            y = y + b  # [B,V,D'] + [1,V,D']
    return y  # [B,V,D']


def multipath_head_aggregation_abae(x, query=None, transform_query=True,
                                    scope='multi_abae', reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        if query is None:
            mu = tf.reduce_mean(x, axis=2)  # [B,V,H,D]->[B,V,D]
        else:
            mu = query  # [B,V,D]
        if transform_query:
            mu = mu + multipath_fully_connected(mu)  # [B,V,D]
        wg = tf.matmul(x, tf.expand_dims(mu, axis=-1))  # [B,V,H,D]x[B,V,D,1]->[B,V,H,1]
        wg = tf.nn.softmax(wg, 2)  # [B,V,H,1] along H
        y = tf.reduce_mean(x * wg, axis=2)  # [B,V,H,D]->[B,V,D]
    return y


def compute_prob_item_given_queries(itm_emb,  # [B,T,D]
                                    itm_msk,  # [B,T], tf.int64, mask if equal zero
                                    query_emb,  # [B,Q,D]
                                    query_msk,  # [B,Q], tf.int64, mask if equal zero
                                    transform_query=True, temperature=0.07,
                                    partitioner=None, scope='prob_item_given_query', reuse=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        _, num_query, emb_dim = get_shape(query_emb)
        _, num_itm, __ = get_shape(itm_emb)
        if transform_query:
            query_emb = query_emb + fully_connected(query_emb, emb_dim, None)  # [B,Q,D]

        prob_item = tf.matmul(query_emb, itm_emb, transpose_b=True)  # [B,Q,D]x[B,T,D]^t->[B,Q,T]
        attn_mask = tf.tile(tf.expand_dims(tf.to_float(tf.not_equal(itm_msk, 0)), axis=1),
                            [1, num_query, 1])  # [B,T]->[B,Q,T]
        prob_item = weighted_softmax(prob_item / temperature, attn_mask, 2)  # 【B,Q,T]

        query_cnt = tf.reduce_sum(query_msk, -1, True)  # [B,1]
        query_cnt = tf.tile(tf.to_float(query_cnt) + 1e-8, [1, num_itm])  # [B,T]
        query_msk = tf.tile(tf.expand_dims(query_msk, axis=2), [1, 1, num_itm])  # [B,Q]->[B,Q,T]
        prob_item = tf.where(tf.equal(query_msk, 0), tf.zeros_like(prob_item), prob_item)  # [B,Q,T]
        prob_item = tf.reduce_sum(prob_item, 1)  # [B,Q,T]->[B,T]
        prob_item = tf.div(prob_item, query_cnt)  # sum(p(item)) = 1
    return prob_item


def compute_prob_item(itm_emb,  # [B,T,D]
                      itm_msk,  # [B,T], tf.int64, mask if equal zero
                      ctx_emb,  # [B,T,D']
                      num_hidden_units,
                      position_bias=False,  # better diversity when no position bias
                      partitioner=None, scope='', reuse=None):
    if position_bias:
        scope += 'prob_itm_psn_ctx'
    else:
        scope += 'prob_itm_ctx'
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        batch_size, seq_len, emb_dim = get_shape(itm_emb)
        if position_bias:
            position_emb = get_dnn_variable(name='position_emb', shape=[1, seq_len, emb_dim],
                                            initializer=get_unit_emb_initializer(emb_dim))
            position_emb = tf.tile(position_emb, [batch_size, 1, 1])
            itm_and_ctx_emb = tf.concat([itm_emb, position_emb, ctx_emb], 2)  # [B,T,D+D+D']
        else:
            itm_and_ctx_emb = tf.concat([itm_emb, ctx_emb], 2)  # [B,T,D+D']
        prob_item = fully_connected(itm_and_ctx_emb, num_hidden_units, None)  # [B,T,D'']
        prob_item = tf.nn.tanh(prob_item)
        prob_item = fully_connected(prob_item, 1, None)  # [B,T,1]
        prob_item = tf.squeeze(prob_item, [2])  # [B,T]
        prob_item = weighted_softmax(
            prob_item, tf.to_float(tf.not_equal(itm_msk, 0)), -1)  # [B,T]
    return prob_item


class ProtoRouter(object):
    def __init__(self, num_proto, emb_dim, temperature=0.07,
                 scope='proto_router', reuse=None, partitioner=None):
        self.num_proto = num_proto
        self.emb_dim = emb_dim
        self.temperature = temperature
        self.proto_embs = get_dnn_variable(
            name='proto_embs_%d' % self.num_proto, shape=[self.num_proto, self.emb_dim],
            partitioner=partitioner, initializer=get_unit_emb_initializer(self.emb_dim),
            scope=scope, reuse=reuse)
        self.proto_embs = tf.nn.l2_normalize(self.proto_embs, -1)  # [H,D]

    def compute_prob_proto(self, x, take_log=False, hard_value=False, soft_grad=True):
        assert len(get_shape(x)) == 2  # only [B,D] is supported
        x = tf.nn.l2_normalize(x, -1)
        y = tf.reduce_sum(tf.multiply(  # [B,1,D]*[1,H,D] -> [B,H,D] -> [B,H]
            tf.expand_dims(x, 1), tf.expand_dims(self.proto_embs, 0)), -1)
        if take_log:
            assert (not hard_value) and soft_grad
            y = tf.nn.log_softmax(y / self.temperature, 1)  # [B,H]
            return y
        y = tf.nn.softmax(y / self.temperature, 1)  # [B,H]
        if hard_value:
            y_hard = tf.one_hot(
                tf.argmax(y, -1), self.num_proto, dtype=tf.float32)
            if soft_grad:
                y = tf.stop_gradient(y_hard - y) + y
            else:
                y = tf.stop_gradient(y_hard)
        else:
            assert soft_grad
        return y  # [B,H]


def disentangle_layer(itm_emb,  # [B,T,D]
                      itm_msk,  # [B,T], tf.int64, mask if equal zero
                      prob_item,  # [B,T]
                      num_heads,
                      proto_router=None,
                      add_head_prior=True,
                      equalize_heads=False,
                      scope='disentangle_layer', reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        batch_size, max_seq_len, emb_dim = get_shape(itm_emb)

        if proto_router is None:
            proto_router = ProtoRouter(num_proto=num_heads, emb_dim=emb_dim)
        assert proto_router.num_proto == num_heads

        prob_head_given_item = proto_router.compute_prob_proto(
            tf.reshape(itm_emb, [batch_size * max_seq_len, emb_dim]))  # [B*T,H]
        prob_head_given_item = tf.reshape(
            prob_head_given_item, [batch_size, max_seq_len, num_heads])  # [B,T,H]
        prob_head_given_item = tf.transpose(prob_head_given_item, [0, 2, 1])  # [B,H,T]

        # p(head, item) = p(item) * p(head | item)
        prob_item_and_head = tf.multiply(  # [B,1,T]*[B,H,T]->[B,H,T]
            tf.expand_dims(prob_item, axis=1), prob_head_given_item)
        itm_msk_expand = tf.tile(tf.expand_dims(itm_msk, 1), [1, num_heads, 1])  # [B,T]->[B,H,T]
        prob_item_and_head = tf.where(
            tf.equal(itm_msk_expand, 0), tf.zeros_like(prob_item_and_head), prob_item_and_head)

        if equalize_heads:
            # p(item | head) = p(head, item) / p(head)
            # Would it be too sensitive/responsive to heads with ONE trigger?
            prob_item_given_head = tf.div(
                prob_item_and_head, tf.reduce_sum(prob_item_and_head, -1, True) + 1e-8)  # [B,H,T]
            init_multi_emb = tf.matmul(prob_item_given_head, tf.nn.l2_normalize(itm_emb, -1))  # [B,H,D]
        else:
            init_multi_emb = tf.matmul(prob_item_and_head, tf.nn.l2_normalize(itm_emb, -1))  # [B,H,D]

        #
        # Spill-over: If no items under this head's category is present in the
        # sequence, the head's vector will mainly be composed of items (with
        # small values of prob_head_given_item for this head) from other
        # categories. As a result, its kNNs will be items from other categories.
        # This effect is sometimes useful, though, since it reuses the empty
        # heads to retrieve relevant items from other categories.
        #
        # Adding a head-specific bias to avoid the spill-over effect when the
        # head is in fact empty. But then the retrieved kNNs may be too
        # irrelevant to the user and make the user unhappy about the result.
        #

        if add_head_prior:
            head_bias = get_dnn_variable(
                name='head_bias', shape=[1, num_heads, emb_dim],
                initializer=get_unit_emb_initializer(emb_dim))
            head_bias = tf.tile(head_bias, [batch_size, 1, 1])  # [B,H,D]
            out_multi_emb = tf.concat([init_multi_emb, head_bias], 2)  # [B,H,2*D]
        else:
            out_multi_emb = init_multi_emb
        out_multi_emb = tf.nn.tanh(fully_connected(out_multi_emb, emb_dim, None))
        out_multi_emb = fully_connected(out_multi_emb, emb_dim, None)
        out_multi_emb = out_multi_emb + init_multi_emb

        #
        # Don't use multipath_fully_connected if num_heads is large, cuz it is memory consuming.
        #
        # out_multi_emb = init_multi_emb
        # out_multi_emb = tf.nn.tanh(multipath_fully_connected(
        #     out_multi_emb, use_bias=add_head_prior, scope='multi_head_fc1'))
        # out_multi_emb = multipath_fully_connected(
        #     out_multi_emb, use_bias=add_head_prior, scope='multi_head_fc2')
        # out_multi_emb = out_multi_emb + init_multi_emb
    return out_multi_emb  # [B,H,D]


def multi_vector_sequence_encoder(itm_emb,  # [B,T,D]
                                  itm_msk,  # [B,T], tf.int64, mask if equal zero
                                  ctx_emb,  # [B,T,D']
                                  num_heads, num_vectors,
                                  scope='mv_seq_enc', reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        batch_size, _, emb_dim = get_shape(itm_emb)
        prob_item = compute_prob_item(  # [B,T]
            itm_emb=itm_emb, itm_msk=itm_msk, ctx_emb=ctx_emb,
            num_hidden_units=emb_dim * 4,
            reuse=reuse, partitioner=partitioner)
        multi_head_emb = disentangle_layer(  # [B,H,D]
            itm_emb=itm_emb, itm_msk=itm_msk, prob_item=prob_item, num_heads=num_heads,
            add_head_prior=True)  # the added prior may lead to weird or over-popular recommendation

        mean_head_emb = tf.matmul(tf.expand_dims(prob_item, 1),
                                  tf.nn.l2_normalize(itm_emb, -1))  # [B,1,T]x[B,T,D]->[B,1,D]
        mean_head_emb = tf.squeeze(mean_head_emb, [1])  # [B,1,D]->[B,D]
        mean_head_emb = mean_head_emb + fully_connected(mean_head_emb, emb_dim, None)
        mean_head_emb = tf.tile(tf.expand_dims(mean_head_emb, 1), [1, num_vectors, 1])

        multi_vec_emb = tf.reshape(
            multi_head_emb, [batch_size, num_vectors, num_heads // num_vectors, emb_dim])
        multi_vec_emb = multipath_head_aggregation_abae(
            multi_vec_emb, query=mean_head_emb, transform_query=False, scope='head2vec')  # [B,V,H,D]->[B,V,D]
    return multi_vec_emb, multi_head_emb  # [B,V,D], [B,H,D]


def disentangled_multi_vector_loss(multi_vec_emb,  # [B,V,D], normalized
                                   pos_emb,  # [B,D], normalized
                                   neg_emb,  # [Q,D], normalized
                                   multi_head_emb,  # [B,H,D], normalized
                                   prob_h,  # [B,H]
                                   query_msk=None,  # [B], tf.bool, mask if False
                                   neg_weights=None,  # [B,Q], w*exp(lgt)
                                   temperature=0.07, margin=0.0,
                                   model_prob_y_and_v=True,  # converge much faster and auc is much better when True
                                   metric_ops=None, scope='multi_loss',
                                   reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        batch_size, num_vectors, emb_dim = get_shape(multi_vec_emb)
        num_negatives = get_shape(neg_emb)[0]
        _, num_heads, _ = get_shape(multi_head_emb)

        # It is Z = \sum_v \sum_i p(v,i), instead of Z_v = \sum_i p(i|v) for each v.
        # The datum is sampled from the support of all possible pairs of (v,i).
        pos_logits = tf.reduce_sum(
            multi_vec_emb * tf.expand_dims(pos_emb, 1), -1, True)  # [B,V,D]*[B,1,D]->[B,V]->[B,V,1]
        neg_logits = tf.reduce_sum(tf.multiply(
            tf.reshape(multi_vec_emb, [batch_size, num_vectors, 1, emb_dim]),
            tf.reshape(neg_emb, [1, 1, num_negatives, emb_dim])), -1)  # [B,V,Q]
        all_logits = tf.concat([pos_logits - margin, neg_logits], 2)  # [B,V,1+Q]

        if neg_weights is None:
            if model_prob_y_and_v:
                raise NotImplementedError
            log_prob_y_given_v = tf.nn.log_softmax(
                all_logits / temperature, -1)  # [B,V,1+Q]
            log_likelihood_v = log_prob_y_given_v[:, :, 0]  # [B,V]
        else:
            neg_weights_expand = tf.tile(
                tf.expand_dims(neg_weights, 1), [1, num_vectors, 1])  # [B,1,Q]->[B,V,Q]
            weights_of_exp = tf.concat(
                [tf.ones_like(pos_logits), neg_weights_expand], 2)  # [B,V,1],[B,V,Q]->[B,V,1+Q]
            # prob_y_v = prob_y_and_v if model_prob_y_and_v else prob_y_given_v
            prob_y_v = weighted_softmax(
                all_logits / temperature, weights_of_exp,
                [1, 2] if model_prob_y_and_v else -1)  # [B,V,1+Q]
            log_likelihood_v = tf.log(prob_y_v[:, :, 0])  # [B,V]

        prob_v = tf.reduce_sum(
            tf.reshape(prob_h, [batch_size, num_vectors, num_heads // num_vectors]), -1)  # [B,V,H/V]->[B,V]
        log_likelihood = tf.reduce_sum(  # expected log_likelihood
            prob_v * log_likelihood_v, -1)  # [B]
        if query_msk is None:
            nll_loss = -tf.reduce_mean(log_likelihood)
        else:
            log_likelihood = tf.where(query_msk, log_likelihood, tf.zeros_like(log_likelihood))
            num_real_queries = tf.reduce_sum(tf.to_float(query_msk)) + 1e-8
            nll_loss = -tf.reduce_sum(log_likelihood) / num_real_queries

        # Issue 1: There are some dead heads that receive no categories.
        prior_h = 1.0 / tf.to_float(num_heads)
        posterior_h = tf.reduce_mean(prob_h, 0)  # [H]
        # version 1: (an okay version)
        # head_kl_loss = tf.reduce_sum(
        #     prior_h * (tf.log(prior_h) - tf.log(posterior_h)), -1)
        # version 2: (much worse than version 1, min(posterior_h) ~ 1e-5
        # head_kl_loss = tf.reduce_sum(
        #     posterior_h * (tf.log(posterior_h) - tf.log(prior_h)), -1)
        # version 3:
        head_kl_loss = tf.reduce_sum(
            prior_h * tf.nn.relu(tf.log(prior_h) - tf.log(posterior_h)), -1)

        #
        # Issue 2: The same category is assigned to more than one heads.
        #
        max_prob_h = tf.reduce_max(prob_h, -1)
        # version 1:
        # sharpness_loss = -tf.reduce_mean(tf.log(max_prob_h))
        max_prob_h_clip = tf.where(tf.greater(max_prob_h, 0.95), tf.ones_like(max_prob_h), max_prob_h)
        sharpness_loss = -tf.reduce_mean(tf.log(max_prob_h_clip))  # clip, don't be be too aggressive
        # version 2:
        # sharpness_loss = -tf.reduce_mean(max_prob_h * tf.log(max_prob_h))
        # version 3: minimizes the entropy for a skewed distribution (too strong and may lead to NaN)
        # sharpness_loss = tf.reduce_sum(tf.multiply(prob_h, tf.log(prob_h)), -1)  # [B,H]->[B]
        # sharpness_loss = -tf.reduce_mean(sharpness_loss, -1)
        # version 4:
        # prob_h_clip = tf.where(  # max(prob_h) being too close to 1.0 will causes tf.log(0)=NaN
        #     tf.tile(tf.greater(tf.reduce_max(prob_h, -1, True), 0.95), [1, num_heads]),
        #     tf.zeros_like(prob_h), prob_h)
        # sharpness_loss = tf.reduce_sum(tf.multiply(prob_h_clip, tf.log(prob_h + 1e-8)), -1)  # [B,H]->[B]
        # sharpness_loss = -tf.reduce_mean(sharpness_loss, -1)
        #
        # Using -p*log(p) or -log(p) can be viewed as using one of the two different
        # directions of KL between the one-hot distribution and p. The gradient
        # (direction & steepness) of -p*log(p) seems to be nicer.
        #

        # Issue 3: The heads's output vectors are the same.
        semantic_loss = tf.reduce_sum(tf.multiply(  # [B,H]
            multi_head_emb, tf.expand_dims(pos_emb, 1)), -1)  # [B,H,D]*[B,1,D]->[B,H]
        semantic_loss = tf.nn.log_softmax(semantic_loss / temperature, -1)  # [B,H]
        # version 1:
        # one_hot_h = tf.one_hot(tf.argmax(prob_h, -1), num_heads, dtype=tf.float32)  # [B,H]
        # semantic_loss = -tf.reduce_mean(
        #     tf.reduce_sum(tf.multiply(semantic_loss, one_hot_h), -1), -1)
        # version 2:
        semantic_loss = -tf.reduce_mean(
            tf.reduce_sum(tf.multiply(semantic_loss, prob_h), -1), -1)

        # Here sharpness_loss and semantic_loss can in fact be unified into one
        # single regularization loss, by using prob_h to weight the latter.

        if metric_ops is not None:
            max_pos_lgt = tf.reduce_max(tf.squeeze(pos_logits, [2]), -1)
            if neg_weights is None:
                max_neg_lgt = tf.reduce_max(tf.reduce_max(neg_logits, -1), -1)  # [B,V,Q]->[B]
            else:
                max_neg_lgt = tf.reduce_max(tf.reduce_max(
                    tf.log(neg_weights_expand + 1e-8) + neg_logits, -1), -1)  # [B,V,Q]->[B]
            add_or_fail_if_exist(metric_ops, scope + '/nll_loss', nll_loss)
            add_accuracy_metrics(metric_ops=metric_ops, scope=scope,
                                 max_pos_lgt=max_pos_lgt, max_neg_lgt=max_neg_lgt,
                                 query_msk=query_msk, neg_weights=neg_weights)
            add_or_fail_if_exist(metric_ops, scope + '_ex/kl_loss', head_kl_loss)
            add_or_fail_if_exist(metric_ops, scope + '_ex/sharp_loss', sharpness_loss)
            add_or_fail_if_exist(metric_ops, scope + '_ex/semantic_loss', semantic_loss)
            add_or_fail_if_exist(metric_ops, scope + '_ex/max_prob_h',
                                 tf.reduce_mean(tf.reduce_max(prob_h, -1)))
            add_or_fail_if_exist(metric_ops, scope + '_ex/max_post_h', tf.reduce_max(posterior_h))
            add_or_fail_if_exist(metric_ops, scope + '_ex/min_post_h', tf.reduce_min(posterior_h))

        aux_loss = head_kl_loss + sharpness_loss + semantic_loss
    return nll_loss, aux_loss
