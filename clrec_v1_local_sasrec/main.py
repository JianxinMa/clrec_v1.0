from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import load_data, evaluate, BatchSampler
from model import Model

FLAGS = tf.app.flags
FLAGS.DEFINE_integer('seed', -1, '')
FLAGS.DEFINE_string('dataset', 'beauty', 'steam/beauty/video/ml-1m/ml-20m')
FLAGS.DEFINE_integer('batch_size', 128, '')
FLAGS.DEFINE_float('lr', 0.001, '')
FLAGS.DEFINE_integer('maxlen', 50, '')
FLAGS.DEFINE_integer('hidden_units', 50, '')
FLAGS.DEFINE_integer('num_blocks', 2, '')
FLAGS.DEFINE_integer('num_epochs', 100, '')
FLAGS.DEFINE_integer('num_heads', 5, '')
FLAGS.DEFINE_float('dropout_rate', 0.5, '')
FLAGS.DEFINE_float('l2_emb', 0.0, '')
FLAGS = FLAGS.FLAGS


def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def main(_):
    if FLAGS.seed == -1:
        FLAGS.seed = np.random.randint(int(2e9))
        print('seed=%d' % FLAGS.seed)
    set_rng_seed(FLAGS.seed)

    print('[ config ] ', ' '.join(
        ['--%s=%s' % (k, v) for k, v in FLAGS.flag_values_dict().items()]))

    dataset = load_data(FLAGS.dataset)
    [user_train, _, _, usernum, itemnum] = dataset

    num_clicks = 0.0
    for u in user_train:
        num_clicks += len(user_train[u])
    print('average sequence length: %.2f' % (num_clicks / len(user_train)))
    num_batch = 2 * int(num_clicks / FLAGS.batch_size) + 1

    sampler = BatchSampler(user_train, usernum, itemnum,
                           batch_size=FLAGS.batch_size, maxlen=FLAGS.maxlen,
                           n_workers=7)
    model = Model(usernum, itemnum, FLAGS)

    print('All global variables:')
    for v in tf.global_variables():
        if v in tf.trainable_variables():
            print('\t', v, 'trainable')
        # else:
        #     print('\t', v)

    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True
    sess_conf.allow_soft_placement = True
    # hooks = [tf.train.ProfilerHook(save_steps=10, output_dir='.')]
    hooks = None
    with tf.train.MonitoredTrainingSession(
            config=sess_conf, hooks=hooks) as sess:
        total_time = 0.0
        t0 = time.time()
        for epoch in range(1, FLAGS.num_epochs + 1):
            for _ in tqdm(range(num_batch), total=num_batch, ncols=70,
                          leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                loss, _ = sess.run([model.loss, model.train_op],
                                   {model.u: u, model.input_seq: seq,
                                    model.pos: pos, model.neg: neg,
                                    model.is_training: True})
                assert not np.isnan(loss)
                assert not np.isinf(loss)
            if epoch % 1 == 0:
                t1 = time.time() - t0
                total_time += t1
                print('epoch: %d, time: %f(s)' % (epoch, total_time))
                t_val = evaluate(model, dataset, FLAGS, sess, testing=False)
                print('val (HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, '
                      'NDCG@5: %.4f, NDCG@10: %.4f, MRR: %.4f)' % t_val)
                t_tst = evaluate(model, dataset, FLAGS, sess, testing=True)
                print('tst (HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, '
                      'NDCG@5: %.4f, NDCG@10: %.4f, MRR: %.4f)' % t_tst)
                print()  # tqdm may overwritten this line
                t0 = time.time()
    sampler.close()
    print("Done")


if __name__ == '__main__':
    tf.app.run()
