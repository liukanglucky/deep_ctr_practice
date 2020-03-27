#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: predict.py
@time: 2019/12/12 上午11:51
@desc:
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')

import numpy as np
import tensorflow as tf
from deep.utils import *
from deep.din_estimator.input_fn import *

batch_size = 1024
model_path = ''
imported_model = tf.saved_model.load(model_path)
eval_data = ""


def original_input_fn(files_path: list, batch_size):
    """
    input_fn
    :param files_path:
    :param batch_size:
    :return:
    """
    data = tf.data.TFRecordDataset(files_path)
    data = data.batch(batch_size)
    return data


def batch_predict(data):
    f = imported_model.signatures["predict"]
    return f(examples=data)["logistic"]


def process():
    eval_files = ["hdfs://xxxx" + p for p in get_file_list(root_path=eval_data)]

    dataset = original_input_fn(eval_files, batch_size=batch_size)
    for d in dataset.take(1):
        batch_predict(d)


if __name__ == '__main__':
    process()
