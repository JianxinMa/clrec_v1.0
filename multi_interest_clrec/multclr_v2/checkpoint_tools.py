# coding=utf-8
from __future__ import division
from __future__ import print_function

import pprint

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.ops import variable_scope


def scan_checkpoint_for_vars(checkpoint_path, vars_to_check):
    check_var_list = checkpoint_utils.list_variables(checkpoint_path)
    # print('check_var_list:', check_var_list)
    # print('vars_to_check:', vars_to_check)
    check_var_set = set()
    for x in check_var_list:
        check_var_set.add(x[0])
    vars_in_checkpoint = []
    vars_not_in_checkpoint = []
    for x in vars_to_check:
        var_name = x.name[:x.name.index(':')]
        if '/part_' in var_name:
            var_name = var_name[:var_name.index('/part_')]
        if var_name in check_var_set:
            vars_in_checkpoint.append(x)
        else:
            vars_not_in_checkpoint.append(x)
    return vars_in_checkpoint, vars_not_in_checkpoint


def assign_from_checkpoint_fn(model_path, var_list,
                              ignore_missing_vars=False,
                              reshape_variables=False):
    if not var_list:
        raise ValueError('var_list cannot be empty')
    if model_path is None:
        return None
    if ignore_missing_vars:
        vars_in_checkpoint, vars_not_in_checkpoint = scan_checkpoint_for_vars(
            model_path, var_list)
        for v in vars_in_checkpoint:
            print('[restoring]', v)
        for v in vars_not_in_checkpoint:
            print('[ignored]', v)
        var_list = vars_in_checkpoint
    if var_list:
        saver = tf.train.Saver(var_list, reshape=reshape_variables,
                               sharded=True)

        def callback(session):
            saver.restore(session, model_path)

        return callback
    else:
        print('No Variables to restore')
        return None


class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_path, include=None, exclude=None):
        self._checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        print('checkpoint_path to restore:', self._checkpoint_path)
        self._include = include
        self._exclude = exclude
        self.init_fn = None

    def begin(self):
        var_list = tf.contrib.framework.get_variables_to_restore(
            include=self._include, exclude=self._exclude)

        self.init_fn = assign_from_checkpoint_fn(
            self._checkpoint_path, var_list=var_list,
            ignore_missing_vars=True, reshape_variables=True)

    def after_create_session(self, session, coord=None):
        # delay until AFTER graph is created
        if session.run(tf.train.get_or_create_global_step()) == 0:
            if self.init_fn:
                print("RestoreHook init_fn...")
                self.init_fn(session)
                print("RestoreHook init_fn done")


def _collect_partitioned_variable(name, var_scope):
    if name + "/part_0" in var_scope._vars:
        var = []
        i = 0
        while name + "/part_%d" % i in var_scope._vars:
            var.append(var_scope._vars[name + "/part_%d" % i])
            i += 1
        return var
    return None


def check_if_variable(current_var_or_name):
    """ Check if this variable is in var_store. """
    var_scope = variable_scope._get_default_variable_store()
    var = var_scope._vars.get(current_var_or_name, None)
    if var is None:
        var = _collect_partitioned_variable(current_var_or_name, var_scope)
    return var


def restore_from_checkpoint(chkpt_dir_or_file, restore_chkpt_scopes=None, chkpt_to_graph_scope_map=None):
    if chkpt_to_graph_scope_map:
        print("map graph scopes to checkpoint scopes: {}".format(chkpt_to_graph_scope_map))

    print("list variables in checkpoint:")
    list_of_name_and_shape = tf.train.list_variables(chkpt_dir_or_file)
    init_assignment_map = dict()
    for chkpt_var_name, chkpt_shape in list_of_name_and_shape:
        if '/Adam' in str(chkpt_var_name):
            continue
        if chkpt_to_graph_scope_map is None or len(chkpt_to_graph_scope_map) == 0:
            graph_var_name = chkpt_var_name
        else:
            graph_var_name = None
            for c_scope, g_scope in chkpt_to_graph_scope_map.items():
                if str(chkpt_var_name).startswith(c_scope):
                    graph_var_name = str(chkpt_var_name).replace(c_scope, g_scope)
                    break
            if graph_var_name is None:
                continue
        graph_var = check_if_variable(graph_var_name)
        graph_shape = None
        if graph_var is not None:
            if not isinstance(graph_var, list):
                graph_shape = graph_var.get_shape().as_list()
            else:
                graph_shape = graph_var[0].get_shape().as_list()
                for i in range(1, len(graph_var)):
                    graph_shape[0] += graph_var[i].get_shape().as_list()[0]
        if graph_var is None:
            print("var not found in graph: name={} checkpoint_name={} checkpoint_shape={}".format(
                graph_var_name, chkpt_var_name, chkpt_shape))
        elif graph_shape != chkpt_shape:
            print("bad shape: name={} checkpoint_name={} shape={} checkpoint_shape={} var={}".format(
                graph_var_name, chkpt_var_name, graph_shape, chkpt_shape, graph_var))
        else:
            print("ready to load: name={} checkpoint_name={} shape={} checkpoint_shape={} var={}".format(
                graph_var_name, chkpt_var_name, graph_shape, chkpt_shape, graph_var))
            init_assignment_map[chkpt_var_name] = graph_var_name

    assignment_map = dict()
    if restore_chkpt_scopes is not None and len(restore_chkpt_scopes) > 0:
        print("load var in checkpoint scope: {}".format(", ".join(restore_chkpt_scopes)))
        for k, v in init_assignment_map.items():
            for var_scope in restore_chkpt_scopes:
                if k.startswith(var_scope):
                    assignment_map[k] = v
    else:
        print('load all possible var from checkpoint')
        assignment_map = init_assignment_map

    print("init_from_checkpoint: " + pprint.pformat(assignment_map))
    checkpoint_utils.init_from_checkpoint(chkpt_dir_or_file, assignment_map)
