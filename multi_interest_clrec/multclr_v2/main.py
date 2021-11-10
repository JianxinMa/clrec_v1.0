# coding=utf-8
from __future__ import division
from __future__ import print_function

import datetime
import os
import sys
import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.client import timeline

from base_model import BaseModel
from checkpoint_tools import RestoreHook
from checkpoint_tools import restore_from_checkpoint
from common_utils import FLAGS
from common_utils import build_input_vars
from common_utils import get_shape


def main(_=None):
    # TF will infer the batch_size and make it a constant if shape_optimization
    # is enabled, which is incompatible with online serving.
    tf.get_default_graph().set_shape_optimize(False)
    print("set shape optimize: False")

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    assert len(worker_hosts) > 0

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    protocol = None
    server_config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    server_config.gpu_options.allow_growth = True
    server_config.allow_soft_placement = True
    server_config.inter_op_parallelism_threads = 20
    server_config.intra_op_parallelism_threads = 20

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             protocol=protocol,
                             config=server_config)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        print('mode:', FLAGS.mode)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            if FLAGS.mode == 'train':
                num_workers = len(FLAGS.worker_hosts.split(','))
                enable_validation = (len(FLAGS.tables.split(',')) >= 2) and (num_workers >= 2)
                is_worker_for_validation = ((FLAGS.task_index + 1) == num_workers)
                input_vars = build_input_vars(repeat_for_epochs=True)
                if enable_validation and is_worker_for_validation:
                    validate_auc_on_small_data(sess_config=server_config, master=server.target)
                else:
                    train(sess_config=server_config, master=server.target, input_vars=input_vars)
            elif FLAGS.mode == 'predict':
                input_vars = build_input_vars(repeat_for_epochs=False)
                predict(server_config=server_config,
                        master=server.target, input_vars=input_vars)
            elif FLAGS.mode == 'auc':
                input_vars = build_input_vars(repeat_for_epochs=False)
                eval_auc_on_small_data(server_config=server_config,
                                       master=server.target, input_vars=input_vars)
            else:
                assert FLAGS.mode == 'export'
                input_vars = build_input_vars(return_placeholders=True)
                partial_restore_and_save(
                    server_config=server_config, master=server.target, input_vars=input_vars)
    print('end')


def train(sess_config, master, input_vars):
    model = BaseModel(input_vars=input_vars, is_training=True)

    for v in tf.global_variables():
        if '/Adam' not in str(v):
            if v in tf.trainable_variables():
                print('  global var:', v, 'trainable')
            else:
                print('  global var:', v, 'frozen')
    for v in tf.local_variables():
        if '/Adam' not in str(v):
            if v in tf.trainable_variables():
                print('  local var:', v, 'trainable')
            else:
                print('  local var:', v, 'frozen')

    archive_chkpt_dir = FLAGS.buckets
    if archive_chkpt_dir == '':
        archive_chkpt_dir = None
    print('init_global_step dir:', archive_chkpt_dir)
    current_chkpt_dir = FLAGS.checkpointDir
    if current_chkpt_dir == '':
        current_chkpt_dir = None
    print('checkpoint dir:', current_chkpt_dir)

    init_global_step = 0
    if archive_chkpt_dir is not None:
        archived_chkpt = tf.train.latest_checkpoint(archive_chkpt_dir)
        if archived_chkpt is not None:
            name = 'model.ckpt-'
            init_global_step = int(archived_chkpt[archived_chkpt.index(name) + len(name):])
    print('init global step:', init_global_step)

    goal_global_step = init_global_step + FLAGS.num_iter
    print('goal global step:', goal_global_step)

    hooks = [tf.train.NanTensorHook(model.loss)]
    chief_only_hooks = []
    if current_chkpt_dir is not None:
        if FLAGS.summary_every > 0:
            chief_only_hooks.append(
                tf.train.SummarySaverHook(
                    save_steps=FLAGS.summary_every, output_dir=current_chkpt_dir, summary_op=model.summary_op))
        if FLAGS.restore_hook:
            assert (archive_chkpt_dir is not None) and (len(archive_chkpt_dir) > 0)
            restore_from_checkpoint(chkpt_dir_or_file=archive_chkpt_dir)
            print('[restore_from_checkpoint] load from buckets:', archive_chkpt_dir)
            chief_only_hooks.append(
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=current_chkpt_dir, save_secs=FLAGS.save_secs, saver=model.saver))
            print('[CheckpointSaverHook] save to checkpointDir:', current_chkpt_dir)

    sys.stdout.flush()
    with tf.train.MonitoredTrainingSession(
            master=master, config=sess_config,
            checkpoint_dir=None if FLAGS.restore_hook else current_chkpt_dir,
            save_checkpoint_secs=None if FLAGS.restore_hook else FLAGS.save_secs,
            chief_only_hooks=chief_only_hooks, is_chief=(FLAGS.task_index == 0),
            hooks=hooks) as mon_sess:
        print("mon_sess started")
        sys.stdout.flush()
        cur_global_step = init_global_step

        now_datetime = str(datetime.datetime.now())
        end_datetime = str(FLAGS.end_time)
        end_datetime = (end_datetime[:4] + '-' + end_datetime[4:6] + '-' + end_datetime[6:8] + ' ' +
                        end_datetime[8:10] + ':' + end_datetime[10:12] + ':' + end_datetime[12:14])
        print('beg_datetime:', now_datetime)
        print('end_datetime:', end_datetime)

        cur_local_step = 0
        last_time = time.time()
        last_global_step = 0
        running_metrics = OrderedDict()
        while (cur_global_step < goal_global_step) and (now_datetime < end_datetime):
            if FLAGS.trace and (FLAGS.task_index == 1) and (current_chkpt_dir is not None) and (
                    cur_local_step % 10000 == 0) and (cur_local_step > 0):
                print("write timeline")
                run_metadata = tf.RunMetadata()
                sess_results = mon_sess.run(
                    model.train_ops,
                    options=tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                tl = timeline.Timeline(run_metadata.step_stats)
                content = tl.generate_chrome_trace_format()
                file_name = 'timeline_' + str(cur_local_step) + '_' + str(
                    FLAGS.task_index) + '.json'
                save_path = os.path.join(current_chkpt_dir, file_name)
                write_gfile = tf.gfile.GFile(save_path, mode='w')
                write_gfile.write(content)
                write_gfile.flush()
                write_gfile.close()
                print("timeline saved to %s" % save_path)
            else:
                sess_results = mon_sess.run(model.train_ops)

            cur_global_step = sess_results['_global_step']
            cur_loss = sess_results['_loss']
            cur_local_step += 1
            for k, v in sess_results.items():
                if (not k.startswith('_')) and (v is not None):
                    if k in running_metrics:
                        running_metrics[k] += v
                    else:
                        running_metrics[k] = v

            if cur_local_step % FLAGS.print_every == 0:
                now_datetime = str(datetime.datetime.now())
                cur_time = time.time()
                local_step_per_sec = FLAGS.print_every / float(cur_time - last_time)
                global_step_per_sec = float(cur_global_step - last_global_step) / (cur_time - last_time)
                hours_to_wait = float(goal_global_step - cur_global_step) / (
                        1e-8 + global_step_per_sec) / 3600.
                print()
                print(now_datetime,
                      'global_step=%d/%d' % (cur_global_step, goal_global_step),
                      'cur_loss=%.5f' % cur_loss,
                      'cur_local_step=%d' % cur_local_step,
                      "global_step_per_sec=%.2f" % global_step_per_sec,
                      "local_step_per_sec=%.2f" % local_step_per_sec,
                      "hours_to_wait=%.2f" % hours_to_wait)

                last_k_prefix = ''
                for k, v in running_metrics.items():
                    if not k.startswith('_'):
                        if not k.startswith(last_k_prefix):
                            print()
                        last_k_prefix = k[:k.index('/') + 1]
                        print('%s=%.6f' % (k, v / float(FLAGS.print_every)), end=' ')

                print()
                sys.stdout.flush()
                last_time = cur_time
                last_global_step = cur_global_step
                running_metrics = OrderedDict()


def validate_auc_on_small_data(sess_config, master):
    from sklearn.metrics import roc_auc_score

    input_data_vars = build_input_vars(repeat_for_epochs=False, mark_serving=False, read_main_table=False)
    input_ph_vars = build_input_vars(return_placeholders=True, mark_serving=False)

    model = BaseModel(input_vars=input_ph_vars)

    archive_chkpt_dir = FLAGS.buckets
    if archive_chkpt_dir == '':
        archive_chkpt_dir = None
    print('init_global_step dir:', archive_chkpt_dir)

    init_global_step = 0
    if archive_chkpt_dir is not None:
        archived_chkpt = tf.train.latest_checkpoint(archive_chkpt_dir)
        if archived_chkpt is not None:
            name = 'model.ckpt-'
            init_global_step = int(archived_chkpt[archived_chkpt.index(name) + len(name):])
    print('init global step:', init_global_step)

    goal_global_step = init_global_step + FLAGS.num_iter
    print('goal global step:', goal_global_step)

    sys.stdout.flush()
    with tf.train.MonitoredTrainingSession(
            master=master, config=sess_config,
            checkpoint_dir=None, save_checkpoint_secs=None, is_chief=False) as mon_sess:
        print("mon_sess started")
        sys.stdout.flush()
        cur_global_step = init_global_step

        samples = []
        num_user_to_eval = 0
        while not mon_sess.should_stop():
            try:
                samples.append(mon_sess.run(dict([(k, v.var) for k, v in input_data_vars.items()])))
                num_user_to_eval += samples[-1]['user__uid'].shape[0]
            except tf.errors.OutOfRangeError:
                print('tf.errors.OutOfRangeError')
                break
            except tf.python_io.OutOfRangeException:
                print('tf.python_io.OutOfRangeException')
                break
        print('num_user_to_eval:', num_user_to_eval)

        while (cur_global_step < goal_global_step) and (not mon_sess.should_stop()):
            all_labels = []
            all_predictions = []
            case_cnt = 0
            global_step_0 = mon_sess.run(model.global_step)
            for case in samples:
                labels, predictions = mon_sess.run(
                    [input_ph_vars['context__label'].var, model.ctr_predictions],
                    feed_dict=dict([(input_ph_vars[k].var, v) for k, v in case.items()]))
                batch_size = labels.shape[0]
                assert batch_size > 0
                assert labels.shape == (batch_size,)
                assert predictions.shape == (batch_size,)
                all_labels.extend([float(x) for x in labels])
                all_predictions.extend([float(x) for x in predictions])
                case_cnt += batch_size
            cur_global_step = mon_sess.run(model.global_step)
            print(datetime.datetime.now(), ' - [global step %d -> global step %d] [%d/%d samples] auc: %.6f' % (
                global_step_0, cur_global_step, case_cnt, num_user_to_eval,
                roc_auc_score(y_true=all_labels, y_score=all_predictions)))
            sys.stdout.flush()


def predict(server_config, master, input_vars):
    writer = OdpsTableWriter(FLAGS.outputs,
                             slice_id=FLAGS.task_index)
    model = BaseModel(input_vars=input_vars)
    if FLAGS.predict_user:
        usr_emb_3d = model.inference_output_3d
    else:
        usr_emb_3d = tf.zeros(shape=(get_shape(model.usr_ids)[0], 1, 1), dtype=tf.float32)
    print('checkpointDir:', FLAGS.checkpointDir)
    sys.stdout.flush()
    assert (FLAGS.checkpointDir is not None) and (len(FLAGS.checkpointDir) > 0)
    with tf.train.MonitoredTrainingSession(
            master=master, config=server_config, is_chief=(FLAGS.task_index == 0),
            checkpoint_dir=FLAGS.checkpointDir, save_checkpoint_secs=None) as mon_sess:
        print(datetime.datetime.now(), "- start mon_sess")
        sys.stdout.flush()
        local_step = 0
        while not mon_sess.should_stop():
            try:
                usr_ids, usr_emb, itm_ids, itm_emb, _ = mon_sess.run(
                    [model.usr_ids, usr_emb_3d,
                     model.pos_nid_ids, model.pos_itm_emb_normalized,
                     model.inc_global_step_op])
                batch_size = usr_ids.shape[0]
                usr_ids = [str(i) for i in usr_ids]
                usr_emb = [';'.join(','.join(str(x) for x in e) for e in u) for u in usr_emb]
                assert len(usr_emb) == batch_size
                itm_ids = [str(i) for i in itm_ids]
                assert len(itm_ids) == batch_size
                itm_emb = [','.join(str(x) for x in e) for e in itm_emb]
                assert len(itm_emb) == batch_size
                writer.write(list(zip(usr_ids, usr_emb, itm_ids, itm_emb)),
                             indices=[0, 1, 2, 3])
                local_step += 1
                if local_step % FLAGS.print_every == 0:
                    print(datetime.datetime.now(),
                          "- %dk cases saved" % (local_step * batch_size // 1000))
                    sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print('tf.errors.OutOfRangeError')
                break
            except tf.python_io.OutOfRangeException:
                print('tf.python_io.OutOfRangeException')
                break
    sys.stdout.flush()
    writer.close()


def eval_auc_on_small_data(server_config, master, input_vars):
    if FLAGS.task_index != 0:
        print('task_index:', FLAGS.task_index)
        print('need only one worker, i.e., the chief worker, since the data is small')
        return

    from sklearn.metrics import roc_auc_score

    model = BaseModel(input_vars=input_vars)
    print('checkpointDir:', FLAGS.checkpointDir)
    sys.stdout.flush()
    assert (FLAGS.checkpointDir is not None) and (len(FLAGS.checkpointDir) > 0)
    all_labels = []
    all_predictions = []
    with tf.train.MonitoredTrainingSession(
            master=master, config=server_config, is_chief=(FLAGS.task_index == 0),
            checkpoint_dir=FLAGS.checkpointDir, save_checkpoint_secs=None) as mon_sess:
        print(datetime.datetime.now(), "- start mon_sess")
        sys.stdout.flush()
        case_cnt = 0
        while not mon_sess.should_stop():
            try:
                labels, predictions, _ = mon_sess.run(
                    [input_vars['context__label'].var, model.ctr_predictions, model.inc_global_step_op])
                batch_size = labels.shape[0]
                assert batch_size > 0
                assert labels.shape == (batch_size,)
                assert predictions.shape == (batch_size,)
                all_labels.extend([float(x) for x in labels])
                all_predictions.extend([float(x) for x in predictions])
                case_cnt += batch_size
                if case_cnt % FLAGS.print_every == 0:
                    print(datetime.datetime.now(), "- %d cases saved" % case_cnt)
                    sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print('tf.errors.OutOfRangeError')
                break
            except tf.python_io.OutOfRangeException:
                print('tf.python_io.OutOfRangeException')
                break
    sys.stdout.flush()
    print('roc_auc_score:', roc_auc_score(y_true=all_labels, y_score=all_predictions))
    sys.stdout.flush()


def partial_restore_and_save(server_config, master, input_vars):
    if FLAGS.task_index != 0:
        print('task_index:', FLAGS.task_index)
        print('need only one worker, i.e., the chief worker')
        return

    model = BaseModel(input_vars=input_vars)

    bucket4load = FLAGS.buckets
    bucket4save = FLAGS.checkpointDir
    print('bucket4load:', bucket4load)
    print('bucket4save:', bucket4save)

    print('restore from checkpoint:', tf.train.latest_checkpoint(bucket4load))
    chief_only_hooks = [RestoreHook(bucket4load)]

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session

    with tf.train.MonitoredTrainingSession(
            master=master,
            config=server_config,
            checkpoint_dir=None,
            save_checkpoint_secs=None,
            chief_only_hooks=chief_only_hooks,
            is_chief=(FLAGS.task_index == 0)) as mon_sess:
        [step] = mon_sess.run([model.global_step])
        print('start from initial global step:', step)
        for _ in range(8):
            [step] = mon_sess.run([model.inc_global_step_op])
        print('incremented the global step to:', step)
        save_path = model.saver.save(get_session(mon_sess),
                                     bucket4save + "model.ckpt",
                                     global_step=model.global_step)
        tf.train.write_graph(
            get_session(mon_sess).graph_def,
            logdir=bucket4save, name='graph.pbtxt', as_text=True)
        print('saved to checkpoint:', save_path)


if __name__ == "__main__":
    tf.app.run()
