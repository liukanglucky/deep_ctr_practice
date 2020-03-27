# -*- coding: UTF-8 -*-
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')

import tensorflow as tf

from tensorflow.python.feature_column.feature_column_v2 import \
    categorical_column_with_vocabulary_list as vocabulary_column
from tensorflow.python.feature_column.feature_column_v2 import crossed_column
from tensorflow.python.feature_column.feature_column_v2 import embedding_column
from tensorflow.python.feature_column.feature_column_v2 import indicator_column

from deep.din_estimator import config

INTEREST_MAX_LEN = config.INTEREST_MAX_LEN

numeric_features = config.CONTINUOUS_COLUMNS

cate_features = config.CATEGORICAL_COLUMNS

hash_features = []

item_ids_features = [

]

user_interest_ids_features = [

]

user_interest_len = [

]

user_neg_interest_ids_features = [

]

user_neg_interest_len = [

]

item_ids_features_vocab_size = {
    "a":1000
}

item_ids_features_emb_size = {
    "a": 10
}

hash_features_bucket_size = 600000
numeric_features_bucket_size = 20


def create_feature_columns():
    age = vocabulary_column('age_level', [c for c in range(1, 7)])
    gender = vocabulary_column('gender', [-1, 1])

    all_cat_cross = crossed_column([age, gender],
                                   hash_bucket_size=100)

    categorical_column = [
        indicator_column(age),
        indicator_column(gender)
    ]

    crossed_columns = [
        indicator_column(all_cat_cross)
    ]

    numerical_column = []

    range_0_20 = [c for c in range(0, 20)]

    embedding_columns = [
        embedding_column(vocabulary_column("order_cnt", range_0_20), dimension=1),
        embedding_column(age, dimension=1),
        embedding_column(gender, dimension=1),
        embedding_column(all_cat_cross, dimension=10)
    ]

    wide_columns = categorical_column + crossed_columns
    deep_columns = numerical_column + embedding_columns
    return wide_columns, deep_columns


def get_feature_description():
    feature_description = {
        "label": tf.io.FixedLenFeature([1], tf.int64)
    }

    # item id
    for f in item_ids_features:
        feature_description[f] = tf.io.FixedLenFeature([1], tf.int64)

    # user interest history
    for f in user_interest_ids_features + user_neg_interest_ids_features:
        feature_description[f] = tf.io.FixedLenFeature([INTEREST_MAX_LEN], tf.int64)

    # user interest history length
    for f in user_interest_len + user_neg_interest_len:
        feature_description[f] = tf.io.FixedLenFeature([1], tf.int64)

    # continuous bucket feature
    for f in numeric_features + cate_features + hash_features:
        feature_description[f] = tf.io.FixedLenFeature([1], tf.int64)

    return feature_description


def input_fn(files_path: list, batch_size):
    """
    input_fn
    :param files_path:
    :param batch_size:
    :return:
    """
    feature_description = get_feature_description()

    all_features = numeric_features + cate_features + hash_features + \
                   item_ids_features + user_interest_ids_features + user_interest_len + \
                   user_neg_interest_ids_features + user_neg_interest_len

    def map_fn(record):
        parsed = tf.io.parse_single_example(record, feature_description)

        feature_map = dict()
        for f in all_features:
            feature_map[f] = parsed[f]

        label = parsed["label"]

        return feature_map, label

    data = tf.data.TFRecordDataset(files_path).map(map_fn, num_parallel_calls=4)
    data = data.shuffle(512)
    data = data.batch(batch_size)
    return data
