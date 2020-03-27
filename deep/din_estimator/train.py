#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: train.py
@time: 2019/12/10 上午10:53
@desc:
"""

import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)])

from deep.din_estimator.deep_interest_network import *
from deep.din_estimator.input_fn import *
from deep.utils import *

model_dir = ""
output_model = ""
train_data = ""
eval_data = ""
train_steps = 1
eval_steps = 1
batch_size = 1024
shuffle_buffer_size = 10000
learning_rate = 0.0003
hidden_units = [128, 80, 40]
attention_hidden_units = [32, 16]
dropout_rate = 0.25
num_parallel_readers = 10
save_checkpoints_steps = 5000
use_batch_norm = True
num_epochs = 3


def main():
    train_files = ["hdfs://xxxx" + p for p in get_file_list(root_path=train_data)]

    eval_files = ["hdfs://xxxx" + p for p in get_file_list(root_path=eval_data)]

    for d in input_fn(train_files, batch_size).take(1):
        print(d)

    print("train_data:", train_files)
    print("eval_data:", eval_files)
    print("train steps:", train_steps, "batch_size:", batch_size)
    print("shuffle_buffer_size:", shuffle_buffer_size)

    wide_columns, deep_columns = create_feature_columns()

    model = DIN(
        params={
            'wide_features': wide_columns,
            'deep_features': deep_columns,
            'hidden_units': hidden_units,
            'learning_rate': learning_rate,
            'attention_hidden_units': attention_hidden_units,
            'vocab_size': item_ids_features_vocab_size,
            'embedding_size': item_ids_features_emb_size,
            'dropout_rate': dropout_rate
        },
        optimizer='Adam',
        config=tf.estimator.RunConfig(model_dir=model_dir,
                                      save_checkpoints_steps=save_checkpoints_steps)
    )

    for i in range(num_epochs):
        print('[INFO: train_and_evalute begin to TRAIN, epoch = ' + str(i) + ']')
        random.shuffle(train_files)

        # early stop
        early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            model,
            eval_dir=model.eval_dir(),
            metric_name='loss',
            max_steps_without_decrease=1000,
            min_steps=100)

        model.train(
            input_fn=
            lambda: input_fn(train_files, batch_size),
            steps=train_steps,
            hooks=[early_stop_hook]
        )

        print('[INFO] train_and_evalute begin to EVALUATE...')
        notice_results = model.evaluate(
            input_fn=lambda: input_fn(eval_files, batch_size),
            steps=eval_steps)
        for key in sorted(notice_results):
            print("[INFO] train_and_evalute == EVALUATE RESULTS == %s: %s" % (
                key, notice_results[key]))

        feature_spec = get_feature_description()
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)
        model.export_saved_model(model_dir + "/saved_model_{0}/".format(i),
                                 serving_input_receiver_fn)


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()
