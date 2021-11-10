from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def min_entropy():
    param = np.random.randn(10, 32)  # [B,H]
    param = tf.Variable(param, trainable=True, dtype=tf.float32)
    prob_h = tf.nn.softmax(param, -1)  # [B,H]

    sharpness_loss = tf.reduce_sum(tf.multiply(prob_h, tf.log(prob_h + 1e-8)), -1)  # [B,H]->[B]
    sharpness_loss = -tf.reduce_mean(sharpness_loss, -1)

    optim_op = tf.train.AdamOptimizer().minimize(sharpness_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        outs = sess.run([optim_op, sharpness_loss, prob_h])
        if i % 100 == 0:
            print(outs[1])
            print(outs[2])


def max_entropy():
    num_heads = 20
    param = np.random.randn(1024, num_heads)  # [B,H]
    param[:, 0] += 10
    param = tf.Variable(param, trainable=True, dtype=tf.float32)
    prob_h = tf.nn.softmax(param, -1)  # [B,H]

    prior_h = 1.0 / tf.to_float(num_heads)
    posterior_h = tf.reduce_mean(prob_h, 0)  # [H]
    head_kl_loss = tf.reduce_sum(
        prior_h * (tf.log(prior_h) - tf.log(posterior_h + 1e-8)), -1)

    optim_op = tf.train.AdamOptimizer().minimize(head_kl_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        outs = sess.run([optim_op, head_kl_loss, posterior_h])
        if i % 100 == 0:
            print(i, outs[1])
            print(outs[2])


if __name__ == '__main__':
    min_entropy()
