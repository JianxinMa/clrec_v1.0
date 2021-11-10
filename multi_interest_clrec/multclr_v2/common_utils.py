# coding=utf-8
from __future__ import division
from __future__ import print_function

import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))

FLAGS = tf.app.flags

FLAGS.DEFINE_integer('task_index', None, '')
FLAGS.DEFINE_string('job_name', '', '')
FLAGS.DEFINE_string('ps_hosts', '', '')
FLAGS.DEFINE_string('worker_hosts', '', '')
FLAGS.DEFINE_string('buckets', None, '')
FLAGS.DEFINE_string('checkpointDir', None, '')
FLAGS.DEFINE_string('target_dir', None, '')
FLAGS.DEFINE_string('tables', '', '')
FLAGS.DEFINE_string('outputs', '', '')

FLAGS.DEFINE_string('end_time', '20771224235900', '')

FLAGS.DEFINE_string('mode', 'train', '')
FLAGS.DEFINE_boolean('restore_hook', False, '')
FLAGS.DEFINE_boolean('predict_user', True, '')
FLAGS.DEFINE_boolean('trace', True, '')
FLAGS.DEFINE_boolean('log_device_placement', False, '')
FLAGS.DEFINE_integer('summary_every', -1, '')
FLAGS.DEFINE_integer('print_every', 200, '')
FLAGS.DEFINE_integer('save_secs', 1800, '')
FLAGS.DEFINE_integer('num_iter', 20000000, '')
FLAGS.DEFINE_integer('num_epoch', 4, '')
FLAGS.DEFINE_integer('batch_size', 128, '')
FLAGS.DEFINE_integer('queue_size', 1280, '')
FLAGS.DEFINE_integer('hard_queue_size', 0, '')
FLAGS.DEFINE_integer('num_heads', 64, '')
FLAGS.DEFINE_integer('num_vectors', 8, '')
FLAGS.DEFINE_float('lr', 0.0001, '')
FLAGS.DEFINE_boolean('rm_dup_neg', True, '')

FLAGS.DEFINE_integer('emb_pt_size', 8 * 1024 * 1024, '')
FLAGS.DEFINE_integer('dnn_pt_size', 64 * 1024, '')
FLAGS.DEFINE_integer('dim', 128, '')
FLAGS.DEFINE_integer('max_len', 256, '')
FLAGS.DEFINE_integer('cut_len', -1, '')

FLAGS.DEFINE_boolean('memorize', True, '')
FLAGS.DEFINE_float('mem_prob', 0.9, '')
FLAGS.DEFINE_integer('num_values_per_slot', 256, '')
FLAGS.DEFINE_float('grad_clip', 0.0, '')

FLAGS = FLAGS.FLAGS


def add_or_fail_if_exist(dictionary, key, val):
    assert key not in dictionary
    dictionary[key] = val


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


if not hasattr(layers, 'fully_connected'):
    print('layers.fully_connected not found')


def fully_connected(inputs, num_outputs, activation_fn):
    return tf.layers.dense(inputs, num_outputs, activation_fn)


_INPUT_VARS_MARKED_FOR_SERVING = set()


class InputVar(object):
    def __init__(self, name, shape, dtype, spec=None):
        if spec is None:
            spec = {}
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.spec = spec
        self.var = None

    def parse_input(self, mark_for_serving, col_to_parse=None):
        if col_to_parse is None:
            var = tf.placeholder(
                dtype=self.dtype, shape=[None if x == -1 else x for x in self.shape], name=self.name)
        else:
            if len(self.shape) == 1:
                var = tf.string_to_number(string_tensor=col_to_parse, out_type=self.dtype)
                var = tf.reshape(var, self.shape)
            else:
                if self.dtype == tf.int64:
                    default_col_val = tf.constant('0', dtype=tf.string)
                else:
                    assert self.dtype == tf.float32
                    default_col_val = tf.constant('0.0', dtype=tf.string)
                assert len(self.shape) == 2
                assert isinstance(self.shape[1], int) and self.shape[1] > 0
                var = tf.decode_csv(
                    records=col_to_parse,  # a tensor of shape [batch_size] with string elements
                    record_defaults=[default_col_val] * self.shape[1],  # feature_num
                    field_delim=',')  # a list whose length is feature_num
                var = tf.stack(var, axis=1)  # [batch_size, feature_num]
                var = tf.string_to_number(var, out_type=self.dtype)
            var = tf.identity(var, name=self.name)

        if (len(self.shape) == 2) and (FLAGS.cut_len > 0) and (FLAGS.max_len > FLAGS.cut_len):
            # PS workers will be the bottleneck if the sequence is very long (too many embedding lookups).
            assert self.shape[1] == FLAGS.max_len
            print('cut_len: %s, %d->%d' % (self.name, FLAGS.max_len, FLAGS.cut_len))
            mask = tf.sequence_mask(
                lengths=tf.ones(shape=[get_shape(var)[0]], dtype=tf.int64) * FLAGS.cut_len,
                maxlen=FLAGS.max_len)
            var = tf.where(mask, var, tf.zeros_like(var))

        self.var = var
        if mark_for_serving:
            assert self.name not in _INPUT_VARS_MARKED_FOR_SERVING
            _INPUT_VARS_MARKED_FOR_SERVING.add(self.name)
            print('inference input: name=%s, tensor=%s' % (self.name, self.var))


def build_hash_fn(h, d=None):
    if d is None:
        return lambda x: tf.mod(x, h) + 1
    return lambda x: tf.mod(tf.div(x, d), h) + 1


def build_bucketize_fn(buckets):
    def bucketize(x):
        x_shape = get_shape(x)
        y = tf.reshape(x, [-1, 1])  # [B,1]
        bkt = tf.constant(value=buckets, dtype=y.dtype)
        bkt = tf.reshape(bkt, [1, -1])  # [1,T]
        y = tf.reduce_sum(tf.to_int64(tf.greater_equal(y, bkt)), -1)  # [B,T] -> [B]
        y = y + 1  # add one because index 0's embedding is zero
        y = tf.reshape(y, x_shape)
        return y

    return bucketize


def build_input_vars(return_placeholders=False, repeat_for_epochs=True, mark_serving=True, read_main_table=True):
    user_hashes = tuple(
        {'table_name': 'user_%d' % i, 'emb_dim': FLAGS.dim // 8,  # smaller dim due to the larger size
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h)} for i, h in enumerate([
            200000051, 200000543, 200002177, 200003539])
    )
    nid_hashes = tuple(
        # Remember to add one when hashing because index 0's embedding is a zero constant.
        {'table_name': 'nid_%d' % i, 'emb_dim': FLAGS.dim // 4,
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h)} for i, h in enumerate([
            80000023, 80000681, 80002243, 80003321])
    )
    seller_hashes = tuple(
        {'table_name': 'seller_%d' % i, 'emb_dim': FLAGS.dim // 4,
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h)} for i, h in enumerate([
            8000009, 8000437, 8002103, 8003111])
    )
    cate_hashes = tuple(
        {'table_name': 'cate_%d' % i, 'emb_dim': FLAGS.dim // 4,
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h)} for i, h in enumerate([
            19081, 19141, 19163, 19207])
    )
    cat1_hashes = tuple(
        {'table_name': 'cat1_%d' % i, 'emb_dim': FLAGS.dim // 4,
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h)} for i, h in enumerate([
            17863, 17891, 17909, 18127])
    )
    abs_time_hashes = tuple(
        {'table_name': 'abs_time_%d' % i, 'emb_dim': FLAGS.dim // 4,
         'bucket_size': h + 8, 'hash_fn': build_hash_fn(h, d)} for i, (d, h) in enumerate([
            # hour of day
            (3600, 24),
            # day of week
            (86400, 7)])
        # day of month
        # (86400, 30),
        # month of year
        # (86400 * 365 // 12, 12),
    )
    rel_time_hashes = tuple(
        {'table_name': 'rel_time_%d' % i, 'emb_dim': FLAGS.dim, 'bucket_size': len(buckets) + 8,
         'hash_fn': build_bucketize_fn(buckets=buckets)} for i, buckets in enumerate([
            (60, 300, 600, 900, 1800, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000,
             39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800,
             86400, 90000, 93600, 97200, 100800, 104400, 108000, 111600, 115200, 118800, 122400, 126000,
             129600, 133200, 136800, 140400, 144000, 147600, 151200, 154800, 158400, 162000, 165600, 169200,
             172800, 259200, 345600, 432000, 518400, 604800, 691200, 777600, 864000, 950400, 1036800, 1123200)])
    )
    stay_hashes = tuple(
        {'table_name': 'st_time_%d' % i, 'emb_dim': FLAGS.dim, 'bucket_size': len(buckets) + 8,
         'hash_fn': build_bucketize_fn(buckets=buckets)} for i, buckets in enumerate([
            (4, 8, 16, 32, 64, 128, 256, 512)])
    )
    input_vars = OrderedDict((x.name, x) for x in [
        InputVar(name='user__uid', shape=(-1,), dtype=tf.int64, spec={'hashes': user_hashes}),
        InputVar(name='user__clk_abs_time', shape=(-1, FLAGS.max_len), dtype=tf.int64,
                 spec={'abs_time_hashes': abs_time_hashes, 'rel_time_hashes': rel_time_hashes}),
        InputVar(name='user__clk_st', shape=(-1, FLAGS.max_len), dtype=tf.int64, spec={'hashes': stay_hashes}),
        InputVar(name='user__clk_nid', shape=(-1, FLAGS.max_len), dtype=tf.int64, spec={'hashes': nid_hashes}),
        InputVar(name='user__clk_uid', shape=(-1, FLAGS.max_len), dtype=tf.int64, spec={'hashes': seller_hashes}),
        InputVar(name='user__clk_cate', shape=(-1, FLAGS.max_len), dtype=tf.int64, spec={'hashes': cate_hashes}),
        InputVar(name='user__clk_cat1', shape=(-1, FLAGS.max_len), dtype=tf.int64, spec={'hashes': cat1_hashes}),
        InputVar(name='user__abs_time', shape=(-1,), dtype=tf.int64,
                 spec={'abs_time_hashes': abs_time_hashes, 'rel_time_hashes': rel_time_hashes}),
        InputVar(name='item__nid', shape=(-1,), dtype=tf.int64, spec={'hashes': nid_hashes}),
        InputVar(name='item__uid', shape=(-1,), dtype=tf.int64, spec={'hashes': seller_hashes}),
        InputVar(name='item__cate', shape=(-1,), dtype=tf.int64, spec={'hashes': cate_hashes}),
        InputVar(name='item__cat1', shape=(-1,), dtype=tf.int64, spec={'hashes': cat1_hashes}),
    ])

    if return_placeholders:
        for name in input_vars:
            input_vars[name].parse_input(mark_for_serving=mark_serving and name.startswith('user__'))
        return input_vars

    if read_main_table:
        # Important: Shuffling is super important for queue-based negative sampling.
        dataset = GlobalShuffledReader(
            [FLAGS.tables.split(',')[0]],
            num_epochs=FLAGS.num_epoch if repeat_for_epochs else 1,
            shuffle=True, num_slices=2000)
        dataset = dataset.input_dataset()
    else:
        dataset = FLAGS.tables.split(',')[1]
    dataset = OdpsTableReader(
        dataset, record_defaults=[''] * len(input_vars),
        selected_cols=','.join(col_name for col_name in input_vars))
    if read_main_table:
        dataset = dataset.batch(FLAGS.batch_size).prefetch(16)
        data_iter = dataset.make_initializable_iterator()
    else:
        dataset = dataset.shuffle(FLAGS.batch_size * 32)
        dataset = dataset.repeat(FLAGS.num_epoch if repeat_for_epochs else 1)
        dataset = dataset.batch(FLAGS.batch_size)
        data_iter = dataset.make_one_shot_iterator()

    # if repeat_infinitely:  # This doesn't work with GlobalShuffledReader.
    #     dataset = dataset.repeat()

    data_cols = data_iter.get_next()
    assert len(data_cols) == len(input_vars)
    for col, name in zip(data_cols, input_vars):
        input_vars[name].parse_input(mark_for_serving=mark_serving and name.startswith('user__'), col_to_parse=col)
    return input_vars
