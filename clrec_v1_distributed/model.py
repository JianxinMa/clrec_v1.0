import numpy as np
import tensorflow as tf

EMB_PT_SIZE = 128 * 1024
DNN_PT_SIZE = 32 * 1024


# Don't use random_uniform_init if you are using the queue-based negative sampling.
# We need to use random_normal instead of random_uniform so that the l2-normalized
# embeddings are evenly distributed on the unit hypersphere.
def get_unit_emb_initializer(emb_sz):
    # initialize emb this way so that the norm is around one
    return tf.random_normal_initializer(mean=0.0, stddev=float(emb_sz ** -0.5))


class NodeEmbedding(object):
    def __init__(self, name, FLAGS, ps_num=None):
        self.node_size = FLAGS.node_count
        self.emb_size = FLAGS.dim
        self.FLAGS = FLAGS
        self.name = name

        self.s2h = FLAGS.s2h

        if FLAGS.ps_hosts:
            self.ps_num = len(FLAGS.ps_hosts.split(","))
        elif ps_num:
            self.ps_num = ps_num

        # self.ps_num = len(FLAGS.ps_hosts.split(","))
        emb_partitioner = tf.min_max_variable_partitioner(max_partitions=self.ps_num, min_slice_size=EMB_PT_SIZE)

        with tf.variable_scope(self.name + '_item_target_embedding', reuse=tf.AUTO_REUSE,
                               partitioner=emb_partitioner) as scope:
            self.emb_table = tf.get_variable("emb_lookup_table", [self.node_size, self.emb_size],
                                             initializer=get_unit_emb_initializer(self.emb_size),
                                             partitioner=emb_partitioner)

        with tf.variable_scope(self.name + '_item_target_bias', reuse=tf.AUTO_REUSE,
                               partitioner=emb_partitioner) as scope:
            self.bias_table = tf.get_variable("bias_lookup_table", [self.node_size], partitioner=emb_partitioner,
                                              initializer=tf.zeros_initializer(), trainable=False)

    def encode(self, index):
        if self.s2h:
            ids = tf.string_to_hash_bucket_fast(index,
                                                self.node_size,
                                                name=self.name + '_to_hash_bucket_oper')
        else:
            ids = tf.string_to_number(index, tf.int64)

        emb = tf.nn.embedding_lookup(self.emb_table,
                                     ids,
                                     name=self.name + '_embedding_lookup_oper_2')

        emb = tf.reshape(emb, [-1, self.emb_size])

        return emb


class SelfAttentive(object):
    def __init__(self,
                 FLAGS,
                 global_step,
                 graph_input,
                 mode='train',
                 ps_num=None):

        self.FLAGS = FLAGS

        self.is_training = True if mode == 'train' else False
        self.global_step = global_step
        self.batch_size = FLAGS.batch_size

        self.s2h = FLAGS.s2h
        self.share_emb = FLAGS.share_emb

        time_bucket = self.FLAGS.time_buckets
        self.time_bucket_size = len(time_bucket.split(",")) * 4

        self.dropout = FLAGS.dropout
        self.hist_max = FLAGS.hist_max
        self.hidden_units = FLAGS.num_hidden_units
        self.num_heads = FLAGS.num_heads
        self.neg_num = FLAGS.neg_num
        self.final_dim = FLAGS.final_dim
        self.dim = FLAGS.dim
        self.encode_depth = FLAGS.encode_depth

        self.nbr_length = graph_input['nbr_mask']
        nbr_mask = tf.sequence_mask(graph_input['nbr_mask'], FLAGS.hist_max)
        self.nbr_mask = tf.to_float(nbr_mask)
        self.ids = graph_input['uid']
        self.i_ids = graph_input['iid']
        self.items = graph_input['item']
        self.samples_mask = graph_input['samples_mask']
        self.tb_feats = graph_input['tb_feats']

        self.item_input_lookup = NodeEmbedding("input", FLAGS, ps_num=ps_num)
        if self.share_emb:
            self.item_output_lookup = self.item_input_lookup
        else:
            self.item_output_lookup = NodeEmbedding("output", FLAGS, ps_num=ps_num)

        with tf.variable_scope('position_embedding', reuse=tf.AUTO_REUSE):
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, self.hist_max, self.dim],
                    name='position_embedding')

        with tf.variable_scope('time_bucket_embedding', reuse=tf.AUTO_REUSE):
            self.tb_embedding = \
                tf.get_variable(
                    shape=[self.time_bucket_size, self.dim],
                    name='time_bucket_embedding')

        if mode == 'export':
            return
        if mode == 'save_emb':
            if FLAGS.user_emb:
                self.seq = self.sequence_encode(self.items, self.nbr_mask, self.tb_feats)
            else:
                self.output_item_emb = self.encode_output_item(self.i_ids)
        else:
            self.loss = self.model()

            with tf.name_scope(mode):
                tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=mode))

            if mode == 'train':
                if FLAGS.learning_algo == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

                self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = [self.opt_op, self.loss, self.global_step]
                self.train_op.extend(self.update_queue_ops)
            else:
                self.eval_op = [self.loss, self.global_step]
            # self.samples = [self.uid, self.iid, self.labels, self.global_step]

        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(sharded=True)

    def encode_input_item(self, input):
        item_list_emb = self.item_input_lookup.encode(tf.reshape(input, [-1]))
        return item_list_emb

    def encode_time_feats(self, tb_feats):
        ids = tf.reshape(tb_feats, [-1])

        emb = tf.nn.embedding_lookup(self.tb_embedding,
                                     ids,
                                     name='tb_embedding_lookup_oper')

        emb = tf.reshape(emb, [-1, self.hist_max, self.dim])

        return emb

    def sequence_encode(self, items, nbr_mask, tb_feats):
        self.item_emb = self.encode_input_item(items)

        tb_emb = self.encode_time_feats(tb_feats)

        item_list_emb = tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding,
                                                    [tf.shape(item_list_emb)[0], 1, 1]) + tb_emb
        # item_list_add_pos = item_list_emb + tb_emb

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units,
                                          activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads,
                                         activation=None, name='fc2')
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            # item_emb = tf.reshape(item_emb, [-1, self.dim * self.num_heads])

            # item_emb = self.normalize(item_emb)

            seq = item_emb

        if self.final_dim > 0:
            with tf.variable_scope("user_final_fc", reuse=tf.AUTO_REUSE):
                # seq = tf.layers.dense(tf.nn.leaky_relu(seq), self.final_dim,
                #                       activation=None, name='proj')
                assert self.final_dim == self.dim
                mu = tf.reduce_mean(seq, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.final_dim, name='maha')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]
                seq = tf.reduce_mean(seq * wg, axis=1)  # [N,H,D]->[N,D]

        seq = tf.nn.l2_normalize(seq, dim=-1)
        return seq

    def normalize(self,
                  inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=tf.AUTO_REUSE):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            # beta = tf.Variable(tf.zeros(params_shape))
            beta = tf.get_variable(
                shape=params_shape,
                name='beta',
                initializer=tf.zeros_initializer())
            # gamma = tf.Variable(tf.ones(params_shape))
            gamma = tf.get_variable(
                shape=params_shape,
                name='gamma',
                initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def encode_output_item(self, input):
        item_emb = self.item_output_lookup.encode(tf.reshape(input, [-1]))

        item_emb = tf.nn.l2_normalize(item_emb, dim=-1)
        return item_emb

    def model(self):
        self.seq = self.sequence_encode(self.items, self.nbr_mask, self.tb_feats)

        loss = self._moco_loss(self.seq)

        return loss

    def _xent_loss(self, user):

        mask_vec_pos = tf.to_float(self.samples_mask)
        emb_dim = self.dim
        if self.final_dim > 0:
            emb_dim = self.final_dim

        assert not self.item_output_lookup.s2h
        i_ids = tf.string_to_number(self.i_ids, tf.int64)
        loss = tf.nn.sampled_softmax_loss(
            weights=self.item_output_lookup.emb_table,
            biases=self.item_output_lookup.bias_table,
            labels=tf.reshape(i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.item_output_lookup.node_size,
            partition_strategy='mod',
            remove_accidental_hits=True,
            exclude_true_classes=True
        )

        loss = tf.reduce_sum(loss * mask_vec_pos) / (
                tf.reduce_sum(mask_vec_pos) + 0.00001)

        return loss

    def _moco_loss(self, user, report_acc=True):
        assert self.FLAGS.share_emb
        assert self.final_dim == self.dim
        emb_dim = self.dim  # C
        queue_size = self.neg_num * self.batch_size  # K

        with tf.variable_scope('moco_queue_scope/%d' % self.FLAGS.task_index,
                               reuse=tf.AUTO_REUSE):
            # self.queue_emb = tf.get_variable(
            #     "moco_queue_emb", trainable=False,
            #     collections=[tf.GraphKeys.LOCAL_VARIABLES],
            #     shape=[queue_size, emb_dim])
            self.queue_idx = tf.get_variable(
                "moco_queue_idx", trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                initializer=tf.constant(
                    [str(int(i)) for i in np.random.choice(
                        self.FLAGS.node_count, size=queue_size)],
                    dtype=tf.string, shape=[queue_size]))

        queue = self.encode_output_item(self.queue_idx)

        # # momentum update: key network
        # momentum = 0.999
        # queue = (momentum * tf.stop_gradient(self.queue_emb)) + (
        #         (1 - momentum) * self.encode_output_item(self.queue_idx))
        # # MoCo uses cosine. Remember to l2-normalize users & items!
        # queue = tf.nn.l2_normalize(queue, dim=-1)

        # q is already l2-normalized in self.sequence_encode.
        q = tf.reshape(user, [-1, emb_dim])  # encoded queries: NxC
        # k is already l2-normalized in self.encode_output_item.
        k = self.encode_output_item(self.i_ids)  # encoded keys: NxC

        # positive logits: Nx1
        n = tf.shape(user)[0]
        l_pos = tf.matmul(
            tf.reshape(q, shape=[n, 1, emb_dim]),
            tf.reshape(k, shape=[n, emb_dim, 1]))
        l_pos = tf.reshape(l_pos, shape=[n, 1])
        # negative logits: NxK
        l_neg = tf.matmul(q, queue, transpose_b=True)
        # logits: Nx(1+K)
        logits = tf.concat([l_pos, l_neg], axis=1)

        # monitor training accuracy
        if report_acc:
            with tf.name_scope('train'):
                pair_acc = tf.reduce_mean(tf.to_float(
                    (l_pos - tf.reduce_max(l_neg, axis=1, keep_dims=True)) > 0))
                tf.summary.scalar('recall_at_top1', pair_acc)

        # contrastive loss
        labels = tf.stop_gradient(
            tf.concat([tf.ones_like(l_pos), tf.zeros_like(l_neg)], axis=1))
        temperature = 0.07
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits / temperature, labels=labels)

        # SGD wrt this loss to update the query network
        mask_vec_pos = tf.to_float(self.samples_mask)
        loss = tf.reduce_sum(loss * mask_vec_pos) / (
                tf.reduce_sum(mask_vec_pos) + 0.00001)

        # update dictionary: enqueue the current & dequeue the earliest batch
        self.update_queue_ops = [
            # self.queue_emb.assign(
            #     tf.concat([queue[n:], k], axis=0)),
            # self.queue_idx.assign(
            #     tf.concat([self.queue_idx[n:], self.i_ids], axis=0))

            # self.queue_idx.assign(
            #     tf.concat([self.queue_idx[n * 2:],
            #                self.i_ids, self.items[:, 0]], axis=0))

            self.queue_idx.assign(
                tf.concat([self.queue_idx[n:], self.i_ids], axis=0))
        ]

        return loss


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(
        initializer=tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32), name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
