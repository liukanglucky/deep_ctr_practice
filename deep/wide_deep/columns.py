#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: columns.py
@time: 2019/11/17 下午5:05
@desc:
"""
import tensorflow as tf

EXAMPLE_VOCABULARY_FILE = ""


def build_columns(embedding_dimension=30):
    # TODO declare your feature_columns here
    # some examples below
    age = tf.feature_column.categorical_column_with_vocabulary_list(
        'age_level', ['-999.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0'])

    day_of_week = tf.feature_column.categorical_column_with_hash_bucket(
        'day_of_week', hash_bucket_size=7)

    poi = tf.feature_column.categorical_column_with_vocabulary_file(
        'poi',
        vocabulary_file=EXAMPLE_VOCABULARY_FILE)

    day_of_week_poi_cross = \
        tf.feature_column.crossed_column(
            ['day_of_week', 'poi'], hash_bucket_size=5000)

    # cross columns
    crossed_columns = [day_of_week_poi_cross]

    categorical_column = [
        tf.feature_column.indicator_column(age),
        day_of_week
    ]

    numeric_column = [
        tf.feature_column.numeric_column('amount', dtype=tf.float64)
    ]

    embedding_columns = [
        tf.feature_column.embedding_column(poi, dimension=embedding_dimension)
    ]

    wide_columns = categorical_column + crossed_columns
    deep_columns = numeric_column + embedding_columns

    # Feature columns for Base
    feature_columns = categorical_column + numeric_column

    return wide_columns, deep_columns, feature_columns
