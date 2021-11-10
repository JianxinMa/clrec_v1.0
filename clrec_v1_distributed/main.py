from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
import samplestore as ss
import tensorflow as tf
from tensorflow.python.client import timeline

from input_graph import GraphInput
from model import SelfAttentive

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

GPU_MEM_FRACTION = 0.8

# Settings
flags = tf.app.flags

flags.DEFINE_integer('hist_max', 20, 'history_max_length')
flags.DEFINE_integer('num_hidden_units', 512, 'attention hidden units')
flags.DEFINE_integer('num_heads', 4, '')
flags.DEFINE_integer('encode_depth', 1, '')

flags.DEFINE_string('time_buckets', '5,10,15,30,60,120,240,480,960,1920,3840', '')

flags.DEFINE_string('checkpointDir', '', '')
flags.DEFINE_string('hdfs_path', '', '')
flags.DEFINE_integer('slice_parts', 30, '')
flags.DEFINE_string('version', '', '')
flags.DEFINE_boolean('use_protobuf_input', False, '')

flags.DEFINE_boolean('s2h', False, '')
flags.DEFINE_boolean('share_emb', True, '')

flags.DEFINE_integer('neg_num', 10, 'negative num')
flags.DEFINE_integer('node_count', 80000100, '')
flags.DEFINE_integer('final_dim', 128, 'Size of final output dim')
flags.DEFINE_integer('shuffle_x', 200, 'x times of batch size for shuffle')

# job param..
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("buckets", None, "oss buckets")
flags.DEFINE_string('mode', 'train', 'train or save_emb')
flags.DEFINE_string('tables', '', 'odps table name')
flags.DEFINE_string('outputs', '', 'odps table name')

flags.DEFINE_boolean('user_emb', False, '')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 12, 'number of epochs to train.')
flags.DEFINE_boolean('trace', False, 'whether trace.')
flags.DEFINE_string('learning_algo', 'adam', 'adam or sgd')
flags.DEFINE_integer('walk_depth', 1, '')
flags.DEFINE_string('job_name', '', 'job name')
flags.DEFINE_boolean('distributed_run', True, 'Whether to use distributed in pai')
flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')

# input structure
flags.DEFINE_integer('dim', 128, 'Size of output dim (final is 2x this, if using concat)')

flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
# logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_integer('validate_iter', 2000, "how often to run a validation minibatch.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 200, "How often to print training info.")
flags.DEFINE_integer('summary_every', 1000000,
                     "How often to print training info.")
flags.DEFINE_integer('save_lines', -1, "How many lines to write to table.")
flags.DEFINE_integer('max_total_steps', 120000000,
                     "Maximum total number of iterations")

# deprecated
flags.DEFINE_integer('python_version', 2, 'python version')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def run_model():
    """
    Initialize session
    """
    mode = FLAGS.mode

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    worker_count = len(worker_hosts)

    # print(ps_hosts)
    # print(worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    protocol = None
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.inter_op_parallelism_threads = 20
    conf.intra_op_parallelism_threads = 20

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             protocol=protocol,  # "grpc++",
                             config=conf)

    # config = OrderedDict(sorted(FLAGS.__flags.items()))
    #
    # if FLAGS.python_version == 3:
    #     for k, v in config.items():
    #         config[k] = v.value
    #
    # print(">>>>> params setting: ")
    # for k, v in config.items():
    #     print(k, config[k])

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        print("mode=", mode)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            print(">>>>> init graph server: ")

            graph_input = GraphInput(FLAGS)
            graph_input.init_server()

            print(">>>>> finish init graph server.")
            """
            create model
            """
            if mode == 'train':
                global_step = tf.train.get_or_create_global_step()
                train(task_id=FLAGS.task_index, worker_count=worker_count, global_step=global_step,
                      master=server.target, graph_input=graph_input, params=FLAGS)
            elif mode == 'save_emb':
                global_step = tf.train.get_or_create_global_step()
                save_emb(task_id=FLAGS.task_index, worker_count=worker_count, global_step=global_step,
                         master=server.target, graph_input=graph_input, params=FLAGS)
            else:
                print("no mode implemented:", mode)
            graph_input.stop()

    print("Finished!")


def train(task_id, worker_count, global_step, master, graph_input, params):
    print("total epoch:", FLAGS.epochs)

    model = SelfAttentive(FLAGS, global_step,
                          graph_input=graph_input.features,
                          mode='train')

    print('All global variables:')
    for v in tf.global_variables():
        if v not in tf.trainable_variables():
            print('\t', v)
        else:
            print('\t', v, 'trainable')

    summary_hook = tf.train.SummarySaverHook(save_steps=FLAGS.summary_every, output_dir=FLAGS.buckets,
                                             summary_op=model.summary_op)

    nan_hook = tf.train.NanTensorHook(model.loss)
    hooks = [nan_hook]
    if FLAGS.max_total_steps != -1:
        stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.max_total_steps)
        hooks.append(stop_hook)

    with tf.train.MonitoredTrainingSession(master=master,
                                           checkpoint_dir=FLAGS.buckets,
                                           save_checkpoint_secs=1500,
                                           chief_only_hooks=[summary_hook],
                                           is_chief=(FLAGS.task_index == 0),
                                           hooks=hooks) as mon_sess:
        print("start training.")
        local_step = 0
        t = time.time()
        last_print_step = 0
        epoch = 0
        while not mon_sess.should_stop():
            try:
                sample_dict = graph_input.feed_next_sample_train()
            except ss.OutOfRangeError:
                epoch += 1
                print('end of an epoch')
                if epoch >= FLAGS.epochs:
                    break
                else:
                    continue

            if FLAGS.trace and FLAGS.task_index == 1 and local_step % 200 == 10 and local_step > 0:
                print("begin to write timeline.")
                run_metadata = tf.RunMetadata()
                outs = mon_sess.run(
                    model.train_op,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata,
                    feed_dict=sample_dict
                )
                tl = timeline.Timeline(run_metadata.step_stats)
                content = tl.generate_chrome_trace_format()
                file_name = 'timeline_' + str(local_step) + '_' + str(FLAGS.task_index) + '.json'
                save_path = os.path.join(FLAGS.buckets, file_name)
                writeGFile = tf.gfile.GFile(save_path, mode='w')
                writeGFile.write(content)
                writeGFile.flush()
                writeGFile.close()
                print("Profiling data save to %s success." % save_path)
            else:
                outs = mon_sess.run(model.train_op, feed_dict=sample_dict)

            train_cost = outs[1]
            global_step = outs[2]

            # Print results
            if local_step % FLAGS.print_every == 0:
                print(datetime.datetime.now(),
                      "type-%d-Iter:" % 0, 'global-%04d' % global_step,
                      'local-%04d' % local_step,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "avg time =", "{:.5f}".format((time.time() - t) * 1.0 / FLAGS.print_every),
                      "global step/sec =", "{:.2f}".format((global_step - last_print_step) * 1.0 / (time.time() - t)))
                t = time.time()
                last_print_step = global_step

            local_step += 1


def save_emb(task_id, worker_count, global_step, master, graph_input, params):
    u_writer = None
    i_writer = None
    if FLAGS.user_emb:
        u_writer = OdpsTableWriter(FLAGS.outputs, slice_id=FLAGS.task_index)
    else:
        i_writer = OdpsTableWriter(FLAGS.outputs, slice_id=FLAGS.task_index)

    model = SelfAttentive(FLAGS, global_step,
                          graph_input=graph_input.features,
                          mode='save_emb')

    max_local_step = -1
    if FLAGS.save_lines != -1:
        max_local_step = FLAGS.save_lines / worker_count
        print('max local step=', max_local_step)

    with tf.train.MonitoredTrainingSession(master=master,
                                           checkpoint_dir=FLAGS.buckets,
                                           is_chief=(FLAGS.task_index == 0),
                                           # log_step_count_steps=None,
                                           save_checkpoint_secs=None
                                           # save_summaries_steps=None,
                                           # save_summaries_secs=None,
                                           ) as mon_sess:
        print("start saving.")
        local_step = 0

        if u_writer is not None:

            while not mon_sess.should_stop():
                try:
                    samples = graph_input.feed_next_user()

                    t = time.time()
                    outs = mon_sess.run([model.ids, model.seq, model.items, model.global_step], feed_dict=samples)
                    # [B,], [B,dim]
                    feat = [','.join(str(x) for x in arr) for arr in outs[1]]
                    input = [','.join(str(x) for x in arr) for arr in outs[2]]
                    u_writer.write(list(zip(outs[0], feat, input)), indices=[0, 1, 2])  # id,emb

                    if local_step % FLAGS.print_every == 0:
                        print("saved", local_step, "emb")
                        print("time cost=", (time.time() - t))
                    local_step += 1

                    if max_local_step != -1 and local_step > max_local_step:
                        break
                except ss.OutOfRangeError:

                    print('user save end of sequence')
                    break
        else:
            while not mon_sess.should_stop():
                try:
                    samples = graph_input.feed_next_item()

                    t = time.time()
                    outs = mon_sess.run([model.i_ids, model.output_item_emb, model.global_step], feed_dict=samples)
                    feat = [','.join(str(x) for x in arr) for arr in outs[1]]
                    i_writer.write(list(zip(outs[0], feat)), indices=[0, 1])

                    if local_step % FLAGS.print_every == 0:
                        print("saved", local_step, "emb")
                        print("time cost=", (time.time() - t))
                    local_step += 1

                    if max_local_step != -1 and local_step > max_local_step:
                        break
                except ss.OutOfRangeError:
                    print('item save end of sequence')

                    break

        if u_writer:
            u_writer.close()
        if i_writer:
            i_writer.close()


def main(argv=None):
    print(FLAGS.mode)
    run_model()


if __name__ == "__main__":
    tf.app.run()
