# coding=utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from common_utils import add_or_fail_if_exist
from common_utils import fully_connected
from common_utils import get_shape


def get_unit_emb_initializer(emb_sz):
    # initialize emb this way so that the norm is around one
    return tf.random_normal_initializer(mean=0.0, stddev=float(emb_sz ** -0.5))


def get_dnn_variable(name, shape, partitioner=None, initializer=None,
                     scope='dnn_param', reuse=None):
    with tf.variable_scope(scope, partitioner=partitioner, reuse=reuse):
        var = tf.get_variable(
            name, shape=shape, dtype=tf.float32, initializer=initializer,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                         ops.GraphKeys.MODEL_VARIABLES])
    return var


def embedding_lookup_unique(params, ids, partition_strategy='mod', name=None):
    """Version of embedding_lookup that avoids duplicate lookups.

    This can save communication in the case of repeated ids.
    Same interface as embedding_lookup. Except it supports multi-dimensional
    `ids` which allows to not reshape input/output to fit gather.

    Args:
        params: A list of tensors with the same shape and type, or a
        `PartitionedVariable`. Shape `[index, d1, d2, ...]`.
        ids: A one-dimensional `Tensor` with type `int32` or `int64` containing
        the ids to be looked up in `params`. Shape `[ids1, ids2, ...]`.
        partition_strategy: "mod" or "div", see embedding_lookup for details.
        name: A string, op name.

    Returns:
        A `Tensor` with the same type as the tensors in `params` and dimension
        of `[ids1, ids2, d1, d2, ...]`.

    Raises:
        ValueError: If `params` is empty.
    """
    ids = tf.convert_to_tensor(ids)
    shape = tf.shape(ids)
    ids_flat = tf.reshape(ids, tf.reduce_prod(shape, None, True))
    unique_ids, idx = tf.unique(ids_flat)
    unique_embeddings = tf.nn.embedding_lookup(
        params, unique_ids, partition_strategy=partition_strategy, name=name)

    embeds_flat = tf.gather(unique_embeddings, idx)
    embed_shape = tf.concat(
        [shape, tf.shape(unique_embeddings)[1:]], 0)
    embeds = tf.reshape(embeds_flat, embed_shape)
    embeds.set_shape(ids.get_shape().concatenate(
        unique_embeddings.get_shape()[1:]))
    return embeds


_EMB_VARS_MARKED_FOR_SERVING = set()


def get_emb_variable(name, ids, shape, mark_for_serving=True, zero_pad=True,
                     partitioner=None, initializer=None, scope='emb_table', reuse=None):
    with tf.variable_scope(scope, partitioner=partitioner, reuse=reuse):
        emb_table = tf.get_variable(
            name, shape=shape, dtype=tf.float32, initializer=initializer,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                         ops.GraphKeys.MODEL_VARIABLES])

        embs = embedding_lookup_unique(emb_table, ids, partition_strategy='mod')

    if mark_for_serving:
        assert name not in _EMB_VARS_MARKED_FOR_SERVING
        _EMB_VARS_MARKED_FOR_SERVING.add(name)
        print('  name=%s,' % name)
        print('  emb_table=%s,' % emb_table.name)
        print('  input=%s,' % ids)
        print('  tensor=%s,' % embs)

    if zero_pad:
        zeros = tf.zeros_like(embs)
        expand_inputs = tf.equal(ids, 0)
        for i in range(len(shape) - 1):
            expand_inputs = tf.expand_dims(expand_inputs, -1)
        masks = tf.tile(expand_inputs, [1] * len(ids.get_shape().as_list()) + list(shape[1:]))
        embs = tf.where(masks, zeros, embs)

    return embs


def mish_activation(x):
    return x * tf.tanh(tf.log(1.0 + tf.exp(x)))


def layer_normalize(inputs, epsilon=1e-8, scale_to_unit_norm=False,
                    scope="ln", reuse=None, partitioner=None):
    with tf.variable_scope(scope, reuse=reuse, partitioner=partitioner):
        inputs_shape = get_shape(inputs)
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = get_dnn_variable(name="beta", shape=params_shape,
                                initializer=tf.zeros_initializer(),
                                partitioner=partitioner)
        gamma = get_dnn_variable(name="gamma", shape=params_shape,
                                 initializer=tf.ones_initializer(),
                                 partitioner=partitioner)
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        if scale_to_unit_norm:
            normalized /= (inputs_shape[-1] ** 0.5)
        outputs = gamma * normalized + beta
    return outputs


def deprecated_masked_softmax(logits, mask, axis, take_log=False):
    padding = tf.to_float(tf.ones_like(mask) * (-2 ** 32 + 1))
    logits = tf.where(tf.equal(mask, 0), padding, logits)
    if take_log:
        return tf.nn.log_softmax(logits, axis)
    return tf.nn.softmax(logits, axis)


# Use this when (1) all weights >= 0, (2) at least one weight > 0.
def weighted_softmax(logits, weights_of_exp, axis):
    z = logits - tf.reduce_max(logits, axis, True)
    # Add epsilon, otherwise it will fail if all weights are zero.
    numerator = tf.exp(z) * weights_of_exp + 1e-8
    denominator = tf.reduce_sum(numerator, axis, True)
    softmax = numerator / denominator
    return softmax


def head_aggregation_abae(x, query=None, scope='abae', reuse=None, transform_query=True, temperature=None):
    batch_size, num_heads, emb_dim = get_shape(x)
    with tf.variable_scope(scope, reuse=reuse):
        if query is None:
            mu = tf.reduce_mean(x, axis=1)  # [B,H,D]->[B,D]
        else:
            mu = query  # [B,D]
        if transform_query:
            mu = mu + fully_connected(mu, emb_dim, None)
        wg = tf.matmul(x, tf.expand_dims(mu, axis=-1))  # [B,H,D] x [B,D,1]
        if temperature is not None:
            wg = tf.div(wg, temperature)
        wg = tf.nn.softmax(wg, 1)  # [B,H,1]
        y = tf.reduce_mean(x * wg, axis=1)  # [B,H,D]->[B,D]
    return y


def simplified_multi_head_attention(itm_emb,  # [B,T,D]
                                    itm_msk,  # [B,T], tf.int64, mask if equal zero
                                    ctx_emb,  # [B,T,D]
                                    num_heads,
                                    num_hidden_units,
                                    scope='simple_multi_head_att',
                                    reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        itm_hidden = fully_connected(
            itm_emb + ctx_emb, num_hidden_units, tf.nn.tanh)
        itm_att = fully_connected(
            itm_hidden, num_heads, None)  # [B,T,H]
        itm_att = tf.transpose(itm_att, [0, 2, 1])  # [B,H,T]
        att_msk = tf.tile(tf.expand_dims(itm_msk, axis=1), [1, num_heads, 1])  # [B,H,T]
        att_pad = tf.to_float(tf.ones_like(att_msk) * (-2 ** 32 + 1))
        itm_att = tf.where(tf.equal(att_msk, 0), att_pad, itm_att)
        itm_att = tf.nn.softmax(itm_att)
        seq_multi_emb = tf.matmul(itm_att, itm_emb)
    return seq_multi_emb  # [B,H,D]


def single_vector_sequence_encoder(itm_emb,
                                   itm_msk,
                                   ctx_emb,
                                   num_heads,
                                   num_hidden_units,
                                   scope='sv_seq_enc',
                                   reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        seq_multi_emb = simplified_multi_head_attention(
            itm_emb=itm_emb, itm_msk=itm_msk, ctx_emb=ctx_emb,
            num_heads=num_heads, num_hidden_units=num_hidden_units,
            scope='simple_multi_head_att', reuse=reuse)
        seq_emb = head_aggregation_abae(seq_multi_emb, scope='abae', reuse=reuse)
    return seq_emb


def add_accuracy_metrics(metric_ops, scope, max_pos_lgt, max_neg_lgt, query_msk, neg_weights):
    batch_size = get_shape(max_pos_lgt)[0]
    if query_msk is None:
        query_msk = tf.ones(shape=[batch_size], dtype=tf.bool)
    else:
        add_or_fail_if_exist(
            metric_ops, scope + '/qw', tf.reduce_mean(tf.to_float(query_msk)))
    num_real_queries = tf.reduce_sum(tf.to_float(query_msk)) + 1e-8
    if neg_weights is not None:
        add_or_fail_if_exist(
            metric_ops, scope + '/nw', tf.reduce_mean(neg_weights))
    ones = tf.ones(shape=[batch_size], dtype=tf.float32)
    zeros = tf.zeros(shape=[batch_size], dtype=tf.float32)
    add_or_fail_if_exist(
        metric_ops, scope + '/min_cos',
        tf.reduce_min(tf.where(query_msk, max_pos_lgt, ones)))
    add_or_fail_if_exist(
        metric_ops, scope + '/max_cos',
        tf.reduce_max(tf.where(query_msk, max_pos_lgt, -ones)))
    add_or_fail_if_exist(
        metric_ops, scope + '/avg_cos', tf.reduce_sum(
            tf.where(query_msk, max_pos_lgt, zeros)) / num_real_queries)
    add_or_fail_if_exist(
        metric_ops, scope + '/rk1_acc', tf.reduce_sum(
            tf.where(query_msk, tf.to_float(max_pos_lgt > max_neg_lgt), zeros)) / num_real_queries)


def single_vector_softmax_loss(query_emb,  # [B,D], normalized
                               pos_emb,  # [B,D], normalized
                               neg_emb,  # [Q,D], normalized
                               query_msk=None,  # [B], tf.bool, mask if False
                               neg_weights=None,  # [B,Q], 0.0<=weights<=1.0, sum(weights)>0, for weights*tf.exp(logits)
                               temperature=0.07,  # temperature is critical if embeddings are l2-normalized
                               margin=0.0,
                               metric_ops=None,
                               scope='loss'):
    batch_size, emb_dim = get_shape(query_emb)
    num_negatives = get_shape(neg_emb)[0]

    pos_logits = tf.reduce_sum(tf.multiply(query_emb, pos_emb), -1, True)  # [B,1]
    neg_logits = tf.reduce_sum(tf.multiply(
        tf.reshape(query_emb, [batch_size, 1, emb_dim]),
        tf.reshape(neg_emb, [1, num_negatives, emb_dim])), -1)  # [B,Q]
    all_logits = tf.concat([pos_logits - margin, neg_logits], 1)  # [B,1+Q]

    if neg_weights is None:
        log_prob_y_given_v = tf.nn.log_softmax(
            all_logits / temperature, -1)  # [B,1+Q]
        log_prob_y_given_v = log_prob_y_given_v[:, 0]
    else:
        weights_of_exp = neg_weights  # [B,Q]
        weights_of_exp = tf.concat(
            [tf.ones_like(pos_logits), weights_of_exp], 1)  # [B,1+Q]
        prob_y_given_v = weighted_softmax(
            all_logits / temperature, weights_of_exp, -1)  # [B,1+Q]
        log_prob_y_given_v = tf.log(prob_y_given_v[:, 0])  # [B]
    if query_msk is None:
        nll_loss = -tf.reduce_mean(log_prob_y_given_v)
    else:
        log_prob_y_given_v = tf.where(query_msk, log_prob_y_given_v, tf.zeros_like(log_prob_y_given_v))
        num_real_queries = tf.reduce_sum(tf.to_float(query_msk)) + 1e-8
        nll_loss = -tf.reduce_sum(log_prob_y_given_v) / num_real_queries

    if metric_ops is not None:
        max_pos_lgt = tf.reduce_max(pos_logits, -1)  # [B]
        if neg_weights is None:
            max_neg_lgt = tf.reduce_max(neg_logits, -1)  # [B]
        else:
            max_neg_lgt = tf.reduce_max(tf.log(neg_weights + 1e-8) + neg_logits, -1)
        add_or_fail_if_exist(metric_ops, scope + '/nll_loss', nll_loss)
        add_accuracy_metrics(metric_ops=metric_ops, scope=scope,
                             max_pos_lgt=max_pos_lgt, max_neg_lgt=max_neg_lgt,
                             query_msk=query_msk, neg_weights=neg_weights)

    return nll_loss
