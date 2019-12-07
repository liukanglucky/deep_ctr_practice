#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: deepfm_train.py
@time: 2019/11/17 下午7:18
@desc:
"""
import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')
from deep import utils
from deep.deep_fm.model import *

# todo set feature size here
total_feature = 10000
feature_field = 50


def input_fn(files: list, feature_len, batch_size=1024, perform_shuffle=False):
    """
    input_fn
    :param files:
    :param feature_len:
    :param batch_size:
    :param perform_shuffle:
    :return:
    """

    def map_fn(record):
        feature_description = {
            "trace_id": tf.io.FixedLenFeature([1], tf.string),
            "feature_index": tf.io.FixedLenFeature([feature_len], tf.float32),
            "feature_value": tf.io.FixedLenFeature([feature_len], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64)
        }
        parsed = tf.io.parse_single_example(record, feature_description)
        return parsed["feature_index"], parsed["feature_value"], parsed["label"]

    data = tf.data.TFRecordDataset(files).map(map_fn)
    if perform_shuffle:
        data = data.shuffle(512)
    data = data.batch(batch_size)
    return data


def train(train_set, vaild_set, batch_size=1024):
    dfm_params = {
        'feature_size': total_feature,
        'field_size': feature_field,
        'train_line': 1000000,
        'valid_line': 500000,
        'k': 10,
        'use_fm': True,
        'use_deep': True,
        'dropout_keep_fm': [0.1, 0.1],
        'deep_layers': [128, 64, 1],
        'dropout_keep_deep': [0.1, 0.1, 0.1],
        'epoch': 100,
        'batch_size': batch_size,
        'learning_rate': 0.01,
        'optimizer_type': 'adam',
        'verbose': 1,
        'random_seed': 1234,
        'loss_type': 'logloss',
        'eval_metric': 'auc',
        'l2': 0.01,
        'l2_fm': 0.01,
        'log_dir': './keras_deepFM_output',
        'bestModelPath': './keras_deepFM_output/deepFM_keras.model',
        'greater_is_better': True
    }

    dfm = DeepFM(**dfm_params)
    dfm.fit_on_libsvm(train_set, vaild_set)


def exec():
    batch_size = 1024

    tr_files = [p for p in utils.get_file_list(root_path="")]
    te_files = [p for p in utils.get_file_list(root_path="")]

    train_set = input_fn(tr_files, feature_field, batch_size=batch_size)
    test_set = input_fn(te_files, feature_field, batch_size=batch_size)
    train(train_set, test_set, batch_size=batch_size)


if __name__ == '__main__':
    exec()
