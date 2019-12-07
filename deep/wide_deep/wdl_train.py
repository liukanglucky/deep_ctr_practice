#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: wdl_train.py
@time: 2019/11/17 下午6:53
@desc:
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import subprocess
import random

from deep.wide_deep.columns import *
from deep.wide_deep.config import *
from deep import utils

DEBUG = True

# 可选有：wide，deep，wide_deep
model_type = "wide_deep"

# 存放模型目录（如用于evaluate）
model_dir = "./model/wide_deep"

if DEBUG:
    num_epochs = 3
else:
    num_epochs = 5

batch_size = 1024
learning_rate = 0.01
l1_reg = 0.001
l2_reg = 0.001
loss_type = "log_loss"
optimizer = "Adam"
dropout = '0.5,0.5,0.5'

# 计算优化
num_threads = 16


def input_fn(files, batch_size=32, perform_shuffle=False, separator='\t', has_header=False):
    """
    input_fn 用于tf.estimators
    :param files:
    :param batch_size:
    :param perform_shuffle:
    :param separator:
    :param has_header: csv文件是否包含列名
    :return:
    """

    def get_columns(file):
        cmd = """hadoop fs -cat {0} | head -1""".format(file)
        status, output = subprocess.getstatusoutput(cmd)
        return output.split("\n")[0].split(separator)

    def map_fn(line):
        defaults = []
        for col in all_columns:
            if col in CONTINUOUS_COLUMNS + ['label']:
                defaults.append([0.0])
            else:
                defaults.append(['0'])
        columns = tf.compat.v2.io.decode_csv(line, defaults, separator, use_quote_delim=False)

        feature_map = dict()

        for fea, col in zip(all_columns, columns):
            if fea not in USE_COLUMNS:
                continue
            feature_map[fea] = col
        labels = feature_map['label']

        return feature_map, labels

    if has_header:
        all_columns = get_columns(files[0])
        # 使用.skip() 跳过csv的第一行
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.flat_map(lambda filename: (
            tf.data.TextLineDataset(filename).skip(1).map(map_fn)))
    else:
        all_columns = COLUMNS
        dataset = tf.data.TextLineDataset(files).map(map_fn)

    if perform_shuffle:
        dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    return dataset



# 构建网络
def build_model(model_dir):
    wide_columns, deep_columns, feature_columns = build_columns()
    l2_reg_rate, l1_reg_rate = l2_reg, l1_reg

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir
    )

    print('[INFO] build_model model_type:', model_type)
    if DEBUG:
        for c in wide_columns: print('[DEBUG] build_model == wide_column:', c)
        for c in deep_columns: print('[DEBUG] build_model == deep_column:', c)
    if model_type == "wide":
        model = tf.estimator.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        model = tf.estimator.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 80, 30])
    else:
        model = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer='Ftrl',
            dnn_feature_columns=deep_columns,
            # must be str, 不能使用tf.optimizers定义, 否则在1st epoch以后将报optimizer重定义的错误
            dnn_optimizer=optimizer,
            dnn_hidden_units=[100, 80, 30],
            config=run_config)

    # add metrics
    # input_fn 使用decode_csv读取, labels shape = () 使用 keras_auc 会报错
    # 从estimator源码中扒出 metrics_lib.auc 解决
    def keras_auc(labels, predictions, features):
        auc_metric = tf.keras.metrics.AUC(name="my_auc")
        auc_metric.update_state(y_true=labels, y_pred=predictions['logistic'],
                                sample_weight=features['weight'])
        return {'auc': auc_metric}

    def my_auc(labels, predictions):
        from tensorflow.python.ops import metrics as metrics_lib
        auc = metrics_lib.auc(
            labels=labels, predictions=predictions['logistic'], curve='ROC',
            name='my_auc')
        return {'my_auc': auc}

    model = tf.estimator.add_metrics(model, my_auc)

    return model


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        if len(gpus) < 2:
            # Create 2 virtual GPUs with 1GB memory each
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
                     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    return gpus


# train and evaluate entrance
def train_and_evaluate():
    print("[INFO] train_and_evaluate step in the real entrence of running ...")

    # load data from normalized path
    tr_files = [p for p in utils.get_file_list(root_path="")]
    va_files = [p for p in utils.get_file_list(root_path="")]
    te_files = [p for p in utils.get_file_list(root_path="")]

    if DEBUG:
        print("[DEBUG] train_and_evaluate tr_files:", tr_files)
        print("[DEBUG] train_and_evaluate va_files:", va_files)
        print("[DEBUG] train_and_evaluate te_files:", te_files)
        print("[DEBUG] train_and_evaluate model directory = %s" % model_dir)

    # 多机多卡训练
    tf.debugging.set_log_device_placement(True)
    set_gpu()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    parallel_batch_size = batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        model = build_model(model_dir)

        for i in range(num_epochs):
            print('[INFO: train_and_evalute begin to TRAIN, epoch = ' + str(i) + ']')
            random.shuffle(tr_files)

            # early stop
            early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
                model,
                eval_dir=model.eval_dir(),
                metric_name='loss',
                max_steps_without_decrease=1000,
                min_steps=100)

            model.train(
                input_fn=
                lambda: input_fn(tr_files, parallel_batch_size, perform_shuffle=True,
                                 has_header=True),
                hooks=[early_stop_hook]
            )

            print('[INFO] train_and_evalute begin to EVALUATE...')
            notice_results = model.evaluate(
                input_fn=
                lambda: input_fn(te_files, parallel_batch_size, perform_shuffle=True,
                                 has_header=True),
                steps=100)
            for key in sorted(notice_results):
                print("[INFO] train_and_evalute == EVALUATE RESULTS == %s: %s" % (
                    key, notice_results[key]))


if __name__ == '__main__':
    train_and_evaluate()
