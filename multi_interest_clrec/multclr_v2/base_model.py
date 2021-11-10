# coding=utf-8
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

from base_modules import get_dnn_variable
from base_modules import get_emb_variable
from base_modules import get_unit_emb_initializer
from base_modules import weighted_softmax
from common_utils import FLAGS
from common_utils import add_or_fail_if_exist
from common_utils import fully_connected
from common_utils import get_shape
from multi_vector_modules import ProtoRouter
from multi_vector_modules import disentangled_multi_vector_loss
from multi_vector_modules import multi_vector_sequence_encoder


class BaseModel(object):
    def __init__(self, input_vars, is_training=False):
        self.is_training = is_training
        self.ps_num = len(FLAGS.ps_hosts.split(','))
        assert self.ps_num > 0

        self.global_step = tf.train.get_or_create_global_step()
        self.inc_global_step_op = self.global_step.assign(self.global_step + 1)

        self.inputs = input_vars
        self.metric_ops = OrderedDict()
        self.queue_ops = []

        self.usr_ids = self.inputs['user__uid'].var  # [B], tf.int64
        self.pos_nid_ids = self.inputs['item__nid'].var  # [B], tf.int64

        with tf.variable_scope(
                name_or_scope='base_pre_emb_queue',
                initializer=tf.glorot_normal_initializer(),
                partitioner=None,  # don't partition the queue
                reuse=tf.AUTO_REUSE):
            self._build_pre_emb_queue()

        with tf.variable_scope(
                name_or_scope='base_model',
                initializer=tf.glorot_normal_initializer(),
                partitioner=tf.min_max_variable_partitioner(
                    max_partitions=self.ps_num, min_slice_size=FLAGS.dnn_pt_size),
                reuse=tf.AUTO_REUSE):
            self._build_embedding()
            self._build_encoder()

        with tf.variable_scope(
                name_or_scope='base_post_emb_queue',
                initializer=tf.glorot_normal_initializer(),
                partitioner=None,  # don't partition the queue
                reuse=tf.AUTO_REUSE):
            self._build_post_emb_queue()

        with tf.variable_scope(
                name_or_scope='base_loss',
                initializer=tf.glorot_normal_initializer(),
                partitioner=tf.min_max_variable_partitioner(
                    max_partitions=self.ps_num, min_slice_size=FLAGS.dnn_pt_size),
                reuse=tf.AUTO_REUSE):
            self._build_optimizer()

        self.train_ops = OrderedDict()
        add_or_fail_if_exist(self.train_ops, '_global_step', self.global_step)
        add_or_fail_if_exist(self.train_ops, '_loss', self.loss)
        add_or_fail_if_exist(self.train_ops, '_optim_op', self.optim_op)
        for i, q in enumerate(self.queue_ops):
            add_or_fail_if_exist(self.train_ops, '_queue_op_%d' % i, q)
        for k, v in self.metric_ops.items():
            add_or_fail_if_exist(self.train_ops, k, v)

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=10, sharded=True, allow_empty=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _build_pre_emb_queue(self):
        assert FLAGS.queue_size >= FLAGS.batch_size
        with tf.variable_scope(
                name_or_scope='pre_emb_queue_of_worker_%d' % FLAGS.task_index,  # local to this worker
                partitioner=None):  # don't partition the queue
            def create_id_queue(queue_name, id_var):
                queue = tf.get_local_variable(
                    queue_name, trainable=False,  # not trainable parameters
                    collections=[tf.GraphKeys.LOCAL_VARIABLES],  # place it on the local worker
                    initializer=tf.zeros_initializer(dtype=tf.int64),
                    dtype=tf.int64, shape=[FLAGS.queue_size])
                # update dictionary: dequeue the earliest batch, and enqueue the current batch
                # the indexing operation, i.e., queue[...], will not work if the queue is partitioned
                updated_queue = tf.concat([queue[get_shape(id_var)[0]:], id_var], axis=0)  # [Q-B],[B]->[Q]
                self.queue_ops.append(queue.assign(updated_queue))
                return updated_queue

            # self.neg_usr_uid = create_id_queue('usr_uid_queue', self.inputs['user__uid'].var)

            self.neg_itm_nid = create_id_queue('itm_nid_queue', self.inputs['item__nid'].var)
            self.neg_itm_uid = create_id_queue('itm_uid_queue', self.inputs['item__uid'].var)
            self.neg_itm_cate = create_id_queue('itm_cate_queue', self.inputs['item__cate'].var)
            self.neg_itm_cat1 = create_id_queue('itm_cat1_queue', self.inputs['item__cat1'].var)

    def _build_embedding(self):
        with tf.variable_scope(
                name_or_scope='embedding_block',
                partitioner=tf.min_max_variable_partitioner(
                    max_partitions=self.ps_num, min_slice_size=FLAGS.emb_pt_size)):

            with tf.variable_scope(name_or_scope='bloom_filter'):
                #
                # We need to ensure that the final embeddings are uniformly distributed on a hyper-sphere,
                # after l2-normalization. Otherwise it will have a hard time converging at the beginning.
                # So we need to use random_normal initialization, rather than random_uniform.
                #
                def bloom_filter_emb(ids, hashes, zero_pad=True, mark_for_serving=True):
                    ids_flat = tf.reshape(ids, [-1])
                    e = []
                    for h in hashes:
                        e.append(get_emb_variable(
                            name=h['table_name'],
                            ids=h['hash_fn'](ids_flat),
                            shape=(h['bucket_size'], h['emb_dim']),
                            mark_for_serving=mark_for_serving,
                            initializer=get_unit_emb_initializer(FLAGS.dim)))  # important: use normal, not uniform
                    e = tf.concat(e, axis=1)
                    if len(hashes) == 1 and hashes[0]['emb_dim'] == FLAGS.dim:
                        print('bloom filter w/o fc: [%s]' % hashes[0]['table_name'])
                    else:
                        dnn_name = 'dnn__' + '__'.join(h['table_name'] for h in hashes)
                        dnn_in_dim = sum(h['emb_dim'] for h in hashes)
                        dnn = get_dnn_variable(
                            name=dnn_name, shape=[dnn_in_dim, FLAGS.dim],
                            initializer=tf.glorot_normal_initializer(),  # important: use normal, not uniform
                            partitioner=tf.min_max_variable_partitioner(
                                max_partitions=self.ps_num, min_slice_size=FLAGS.dnn_pt_size))
                        e = tf.matmul(e, dnn)
                    if zero_pad:
                        id_eq_zero = tf.tile(tf.expand_dims(tf.equal(ids_flat, 0), -1), [1, FLAGS.dim])
                        e = tf.where(id_eq_zero, tf.zeros_like(e), e)
                    e = tf.reshape(e, get_shape(ids) + [FLAGS.dim])
                    return e

                def combine_mean(emb_list):
                    assert len(emb_list) >= 2
                    return sum(emb_list) * (1.0 / len(emb_list))

                self.usr_mem_emb = bloom_filter_emb(ids=self.inputs['user__uid'].var,
                                                    hashes=self.inputs['user__uid'].spec['hashes'])
                self.is_recent_click = tf.greater(self.inputs['user__clk_st'].var, 0)  # [B,T], tf.bool
                self.is_recent_click_expand = tf.tile(tf.expand_dims(self.is_recent_click, -1), [1, 1, FLAGS.dim])

                clk_st_emb = bloom_filter_emb(ids=tf.abs(self.inputs['user__clk_st'].var),
                                              hashes=self.inputs['user__clk_st'].spec['hashes'])
                clk_rel_time_emb = bloom_filter_emb(
                    ids=tf.tile(tf.expand_dims(self.inputs['user__abs_time'].var, -1),
                                [1, FLAGS.max_len]) - self.inputs['user__clk_abs_time'].var,
                    hashes=self.inputs['user__clk_abs_time'].spec['rel_time_hashes'])

                clk_nid_emb = bloom_filter_emb(ids=self.inputs['user__clk_nid'].var,
                                               hashes=self.inputs['user__clk_nid'].spec['hashes'])
                clk_uid_emb = bloom_filter_emb(ids=self.inputs['user__clk_uid'].var,
                                               hashes=self.inputs['user__clk_uid'].spec['hashes'])
                clk_cate_emb = bloom_filter_emb(ids=self.inputs['user__clk_cate'].var,
                                                hashes=self.inputs['user__clk_cate'].spec['hashes'])
                clk_cat1_emb = bloom_filter_emb(ids=self.inputs['user__clk_cat1'].var,
                                                hashes=self.inputs['user__clk_cat1'].spec['hashes'])
                self.clk_itm_emb = combine_mean([clk_nid_emb, clk_uid_emb, clk_cate_emb, clk_cat1_emb])  # [B,T,D]

                clk_ctx_time_key = bloom_filter_emb(
                    ids=tf.concat([tf.expand_dims(self.inputs['user__abs_time'].var, -1),
                                   self.inputs['user__clk_abs_time'].var], 1),  # [B,1] [B,T] -> [B,1+T]
                    hashes=self.inputs['user__clk_abs_time'].spec['abs_time_hashes'])  # [B,1+T,D]
                usr_ctx_time_query, clk_ctx_time_key = tf.split(
                    clk_ctx_time_key, [1, FLAGS.max_len], 1)  # [B,1,D], [B,T,D]
                usr_ctx_time_query = tf.squeeze(usr_ctx_time_query, [1])  # [B,D]

                prob_psn = get_dnn_variable(name='position_w_%d' % FLAGS.max_len, shape=[1, FLAGS.max_len],
                                            initializer=get_unit_emb_initializer(FLAGS.dim))
                prob_psn = tf.tile(prob_psn, [get_shape(self.usr_ids)[0], 1])  # [1,T]->[B,T]
                prob_psn = weighted_softmax(prob_psn, tf.to_float(self.is_recent_click), axis=-1)  # [B,T] along T

                clk_ctx_cate_key = combine_mean([clk_cate_emb, clk_cat1_emb])  # [B,T,D]
                usr_ctx_cate_query = tf.squeeze(tf.matmul(
                    tf.expand_dims(prob_psn, 1), clk_ctx_cate_key), [1])  # [B,1,T]x[B,T,D]->[B,1,D]->[B,D]
                self.clk_ctx_key_emb = tf.concat([
                    combine_mean([clk_st_emb, clk_rel_time_emb]), clk_ctx_time_key, clk_ctx_cate_key], 2)  # [B,T,3*D]
                self.usr_ctx_query_emb = tf.concat([usr_ctx_time_query, usr_ctx_cate_query], 1)  # [B,2*D]

                pos_nid_emb = bloom_filter_emb(ids=self.inputs['item__nid'].var,
                                               hashes=self.inputs['item__nid'].spec['hashes'],
                                               mark_for_serving=False)
                pos_uid_emb = bloom_filter_emb(ids=self.inputs['item__uid'].var,
                                               hashes=self.inputs['item__uid'].spec['hashes'],
                                               mark_for_serving=False)
                pos_cate_emb = bloom_filter_emb(ids=self.inputs['item__cate'].var,
                                                hashes=self.inputs['item__cate'].spec['hashes'],
                                                mark_for_serving=False)
                pos_cat1_emb = bloom_filter_emb(ids=self.inputs['item__cat1'].var,
                                                hashes=self.inputs['item__cat1'].spec['hashes'],
                                                mark_for_serving=False)
                self.pos_itm_emb = combine_mean([pos_nid_emb, pos_uid_emb, pos_cate_emb, pos_cat1_emb])  # [B,D]
                self.pos_itm_emb_normalized = tf.nn.l2_normalize(self.pos_itm_emb, -1)
                self.pos_cat_emb = combine_mean([pos_cate_emb, pos_cat1_emb])  # [B,D]

                neg_nid_emb = bloom_filter_emb(ids=self.neg_itm_nid,
                                               hashes=self.inputs['item__nid'].spec['hashes'],
                                               mark_for_serving=False)
                neg_uid_emb = bloom_filter_emb(ids=self.neg_itm_uid,
                                               hashes=self.inputs['item__uid'].spec['hashes'],
                                               mark_for_serving=False)
                neg_cate_emb = bloom_filter_emb(ids=self.neg_itm_cate,
                                                hashes=self.inputs['item__cate'].spec['hashes'],
                                                mark_for_serving=False)
                neg_cat1_emb = bloom_filter_emb(ids=self.neg_itm_cat1,
                                                hashes=self.inputs['item__cat1'].spec['hashes'],
                                                mark_for_serving=False)
                self.neg_itm_emb = combine_mean([neg_nid_emb, neg_uid_emb, neg_cate_emb, neg_cat1_emb])  # [Q,D]
                self.neg_cat_emb = combine_mean([neg_cate_emb, neg_cat1_emb])  # [Q,D]

    def _build_encoder(self):
        with tf.variable_scope(name_or_scope='encoder_block'):
            batch_size = get_shape(self.usr_ids)[0]
            seq_itm_msk = self.inputs['user__clk_nid'].var
            seq_itm_emb = self.clk_itm_emb
            with tf.variable_scope(name_or_scope='ctx_q_as_mlp'):
                usr_ctx_q = fully_connected(self.usr_ctx_query_emb,
                                            FLAGS.dim * FLAGS.dim, None)  # [B,D'']->[B,DxD]
                usr_ctx_q = tf.reshape(usr_ctx_q, [batch_size, FLAGS.dim, FLAGS.dim])  # [B,D,D]
            with tf.variable_scope(name_or_scope='ctx_proj_k'):
                seq_ctx_k = fully_connected(self.clk_ctx_key_emb, FLAGS.dim, None)  # [B,T,D']->[B,T,D]

            def ctx_co_action(q, k):  # [B,D,D], [B,T,D]
                return tf.tanh(k + tf.matmul(tf.tanh(k), q))  # [B,T,D]x[B,D,D]->[B,T,D]

            seq_ctx_qk = ctx_co_action(q=usr_ctx_q, k=seq_ctx_k)  # [B,T,D]

            self.multi_vec_emb, self.multi_head_emb = multi_vector_sequence_encoder(
                itm_emb=seq_itm_emb, itm_msk=seq_itm_msk, ctx_emb=seq_ctx_qk,
                num_heads=FLAGS.num_heads, num_vectors=FLAGS.num_vectors,
                scope='enc_%dh%dv' % (FLAGS.num_heads, FLAGS.num_vectors))  # [B,V,D]
            self.multi_vec_emb_normalized = tf.nn.l2_normalize(self.multi_vec_emb, -1)  # [B,V,D]

            self.inference_output_3d = self.multi_vec_emb_normalized

            with tf.variable_scope("predictions"):
                output_name = 'user_emb'
                inference_output_2d = tf.reshape(self.inference_output_3d, [-1, FLAGS.dim])  # [B*V,D]
                inference_output_2d = tf.identity(inference_output_2d, output_name)
                print('inference output: name=%s, tensor=%s' % (output_name, inference_output_2d))

                # not really ctr, just a score between 0.0 and 1.0
                self.ctr_predictions = tf.matmul(self.inference_output_3d,  # [B,V,D]x[B,D,1]->[B,V,1]
                                                 tf.expand_dims(self.pos_itm_emb_normalized, 2))
                self.ctr_predictions = tf.reduce_max(tf.squeeze(self.ctr_predictions, [2]), -1)  # [B,V]->[B]

    def _build_post_emb_queue(self):
        with tf.variable_scope(
                name_or_scope='post_emb_queue_of_worker_%d' % FLAGS.task_index,
                partitioner=None):
            def create_emb_queue(queue_name, emb_var):
                queue = tf.get_local_variable(
                    queue_name, trainable=False,  # not trainable parameters
                    collections=[tf.GraphKeys.LOCAL_VARIABLES],  # place it on the local worker
                    initializer=get_unit_emb_initializer(FLAGS.dim),
                    dtype=tf.float32, shape=[FLAGS.queue_size, FLAGS.dim])
                # update dictionary: dequeue the earliest batch, and enqueue the current batch
                # the indexing operation, i.e., queue[...], will not work if the queue is partitioned
                updated_queue = tf.concat([queue[get_shape(emb_var)[0]:], emb_var], axis=0)  # [Q-B,D],[B,D]->[Q,D]
                self.queue_ops.append(queue.assign(updated_queue))
                return updated_queue

            if hasattr(self, 'neg_itm_emb'):
                print('itm_emb_queue for caching disabled')
            else:
                self.neg_itm_emb = create_emb_queue("itm_emb_queue", self.pos_itm_emb)

    def _build_optimizer(self):
        with tf.variable_scope(name_or_scope='optimizer_block'):
            batch_size = get_shape(self.pos_nid_ids)[0]
            cate_router = ProtoRouter(num_proto=FLAGS.num_heads, emb_dim=FLAGS.dim)
            pos_prob_h = cate_router.compute_prob_proto(self.pos_cat_emb)  # [B,H]

            # the user and item embeddings need to be l2-normalized when using the contrastive loss
            neg_itm_emb_normalized = tf.nn.l2_normalize(self.neg_itm_emb, -1)
            neg_itm_weights = tf.to_float(tf.not_equal(
                tf.expand_dims(self.pos_nid_ids, -1), tf.expand_dims(self.neg_itm_nid, 0)))  # [B,1]==[1,Q] -> [B,Q]

            if FLAGS.hard_queue_size > 0:
                num_easy_neg = FLAGS.queue_size - FLAGS.hard_queue_size
                assert num_easy_neg >= FLAGS.batch_size
                neg_prob_h = cate_router.compute_prob_proto(self.neg_cat_emb[:FLAGS.hard_queue_size])  # [Q',H]
                hard_neg_msk = tf.to_float(tf.equal(
                    tf.expand_dims(tf.argmax(pos_prob_h, -1), -1),
                    tf.expand_dims(tf.argmax(neg_prob_h, -1), 0)))  # [B,1]==[1,Q']->[B,Q']
                hard_neg_cnt = tf.reduce_sum(hard_neg_msk, -1)  # [B]
                add_or_fail_if_exist(self.metric_ops, 'neg_queue/max_hard_cnt', tf.reduce_max(hard_neg_cnt, -1))
                add_or_fail_if_exist(self.metric_ops, 'neg_queue/min_hard_cnt', tf.reduce_min(hard_neg_cnt, -1))
                hard_neg_msk = tf.concat(  # [B,Q]
                    [hard_neg_msk, tf.ones(shape=(batch_size, num_easy_neg), dtype=tf.float32)], 1)
                neg_itm_weights = tf.multiply(neg_itm_weights, hard_neg_msk)  # [B,Q]

            if FLAGS.rm_dup_neg:
                # This implementation only de-duplicate easy negative samples. Hard ones are not de-duplicated.
                easy_neg_nid = self.neg_itm_nid[FLAGS.hard_queue_size:]
                neg_itm_appear_cnt = tf.reduce_sum(tf.to_float(  # [Q'',1]==[1,Q'']->[Q'',Q'']->[Q'']
                    tf.equal(tf.expand_dims(easy_neg_nid, 1), tf.expand_dims(easy_neg_nid, 0))), -1)
                add_or_fail_if_exist(self.metric_ops, 'neg_queue/max_appear_cnt', tf.reduce_max(neg_itm_appear_cnt))
                add_or_fail_if_exist(self.metric_ops, 'neg_queue/min_appear_cnt', tf.reduce_min(neg_itm_appear_cnt))
                neg_itm_appear_cnt = tf.concat([tf.ones(shape=[FLAGS.hard_queue_size], dtype=tf.float32),
                                                neg_itm_appear_cnt], 0)  # [Q'],[Q'']->[Q]
                neg_itm_weights = tf.div(neg_itm_weights, tf.expand_dims(neg_itm_appear_cnt, 0))  # [B,Q]/[1,Q]

            multi_vec_nll_loss, disentangle_aux_loss = disentangled_multi_vector_loss(
                multi_vec_emb=self.multi_vec_emb_normalized,
                pos_emb=self.pos_itm_emb_normalized,
                neg_emb=neg_itm_emb_normalized,
                multi_head_emb=tf.nn.l2_normalize(self.multi_head_emb, -1),
                prob_h=pos_prob_h,
                neg_weights=neg_itm_weights,
                metric_ops=self.metric_ops,
                scope='multi')
            self.loss = multi_vec_nll_loss
            self.loss += disentangle_aux_loss

            self.optim_op = tf.train.AdamOptimizer(use_locking=True)
            if FLAGS.grad_clip > 1e-3:
                # sources of NaN: (1) div-zero; (2) log(non-positive); (3) gradient explosion; ...
                grads = self.optim_op.compute_gradients(self.loss)
                grads = [(tf.clip_by_norm(g, FLAGS.grad_clip), v) for g, v in grads]
                self.optim_op = self.optim_op.apply_gradients(grads, global_step=self.global_step)
            else:
                self.optim_op = self.optim_op.minimize(self.loss, global_step=self.global_step)
