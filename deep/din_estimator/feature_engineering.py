#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: feature_engineering.py
@time: 2019/12/4 下午2:45
@desc:
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')

import numpy as np
import pandas as pd
import json

from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import feature
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.ml.feature import QuantileDiscretizer

from deep.din_estimator import config
from deep import utils


def _get_data(spark: SparkSession, start_date, end_date):
    sql = config.SAMPLE_SQL_TPL.format(
        cols=','.join(col for col in config.COLUMNS),
        START_DATE=start_date,
        END_DATE=end_date)
    print("[INFO] get data sql: {0}".format(sql))
    return spark.sql(sql).na.fill(-999)


def process(sc: SparkContext,
            train_data: DataFrame, test_data: DataFrame):
    train_data.show(10, False)
    # 1. 连续特征分桶
    train_data, test_data, continuous_feature_bucket_splits_dict = \
        _sparse_feature_with_quantile(train_data, test_data, sc)
    train_data.show(10, False)
    # 2. 高维离散特征hash
    train_data, test_data, hash_high_dimensional_features = \
        _hash_features(config.HIGH_DIMENSIONAL_SPARSE_COLUMNS, train_data, test_data)

    train_data.show(10, False)
    # 4. get interest
    train_data = _get_interest(train_data)
    test_data = _get_interest(test_data)

    train_data.show(10, False)
    # 5. rename
    for f in config.CONTINUOUS_COLUMNS:
        train_data = train_data.drop(f).withColumnRenamed(config.BUCKET_FEATURE_PREFIX + f, f)
        test_data = test_data.drop(f).withColumnRenamed(config.BUCKET_FEATURE_PREFIX + f, f)
    train_data.show(10, False)
    # 触发action
    train_data.groupBy('label').count().show(10, False)
    test_data.groupBy('label').count().show(10, False)

    # 5. save data into table
    save_cols = ["label"] \
                + config.CONTINUOUS_COLUMNS \
                + hash_high_dimensional_features \
                + config.CATEGORICAL_COLUMNS \
                + config.ITEM_ID_COLUMNS \
                + config.INTEREST_COLUMNS \
                + [col + "_len" for col in config.INTEREST_COLUMNS]

    print("save cols : {0}".format(str(save_cols)))

    train_data.select(save_cols).coalesce(500).write.mode('overwrite').format("tfrecords").option(
        "recordType", "Example").save(config.SAMPLE_SAVE_DIR + "/train/")
    test_data.select(save_cols).coalesce(500).write.mode('overwrite').format("tfrecords").option(
        "recordType", "Example").save(config.SAMPLE_SAVE_DIR + "/test/")


def _get_interest(data: DataFrame):
    def _to_interest_history_vector(value: str):
        if value is None or value == "":
            # return Vectors.zeros(config.INTEREST_MAX_LEN)
            return np.zeros(config.INTEREST_MAX_LEN, dtype=int)
        value = np.array(value.split(","))
        # 长度不足左侧补零(只是为了方便矩阵运算，model中attention层加入mask可以忽略补的0)
        if len(value) < config.INTEREST_MAX_LEN:
            value = np.pad(value, (config.INTEREST_MAX_LEN - len(value), 0), 'constant')
        else:
            # 长度过长截断
            value = value[-config.INTEREST_MAX_LEN:]
        # 不能返回numpy.dtype类型数据
        value = [int(v) for v in value]
        return value
        # return Vectors.dense(value)

    def _get_interest_history_length(value: str):
        if value is None or value == "" or value == "0":
            return 0
        value_size = len(value.split(","))
        if value_size > config.INTEREST_MAX_LEN:
            return config.INTEREST_MAX_LEN
        return int(value_size)

    # transform interest history
    to_vector_udf = F.udf(_to_interest_history_vector, ArrayType(IntegerType()))
    # to_vector_udf = F.udf(_to_interest_history_vector, VectorUDT())
    hist_len_udf = F.udf(_get_interest_history_length, IntegerType())
    for col in config.INTEREST_COLUMNS:
        data = data.withColumn(col + "_vec", to_vector_udf(data[col])) \
            .withColumn(col + "_len", hist_len_udf(data[col]).astype("int")) \
            .drop(col).withColumnRenamed(col + "_vec", col)
    return data


def _hash_value_udf(hash_bucket: int):
    def _wrapper(v):
        return int(abs(hash(str(v))) % hash_bucket + 1)

    return F.udf(lambda x: _wrapper(x))


def _hash_features(need_hash_features: dict, train_data: DataFrame, test_data: DataFrame):
    hash_features = list()
    for feature, hash_num in need_hash_features.items():
        new_feature_name = config.HASH_FEATURE_PREFIX + feature
        train_data = train_data.withColumn(
            new_feature_name,
            _hash_value_udf(hash_num)(train_data[feature]).cast("int"))
        test_data = test_data.withColumn(
            new_feature_name,
            _hash_value_udf(hash_num)(test_data[feature]).cast("int"))
        hash_features.append(new_feature_name)
        print("generate new hash features {0}. ".format(new_feature_name))
    return train_data, test_data, hash_features


def _cross_features(need_cross_features: list, train_data: DataFrame, test_data: DataFrame):
    cross_features = list()
    for item in need_cross_features:
        if not isinstance(item, dict) \
                or "feature_list" not in item.keys() \
                or "hash_bucket_size" not in item.keys():
            print("need_cross_features must be a dict "
                  "with key 'feature_list' and 'hash_bucket_size' !")
            continue

        # 连续特征使用 分桶后的数据交叉
        concat_features, hash_num = item["feature_list"], item["hash_bucket_size"]
        new_feature_name = config.HASH_FEATURE_PREFIX + "_".join(concat_features)

        train_data = \
            train_data.withColumn(
                new_feature_name,
                _hash_value_udf(hash_num)(F.concat_ws("_", *concat_features)).cast("int"))

        test_data = \
            test_data.withColumn(
                new_feature_name,
                _hash_value_udf(hash_num)(F.concat_ws("_", *concat_features)).cast("int"))

        cross_features.append(new_feature_name)
        print("generate new crossed features {0}. ".format(new_feature_name))
    return train_data, test_data, cross_features


def _sparse_feature_with_quantile(train_data: DataFrame,
                                  test_data: DataFrame,
                                  sc: SparkContext):
    """
    连续特征离散化
    :param train_data:
    :param test_data:
    :param sc: sparkContext
    :return: 含有离散化特征的DataFrame, 分桶splits
    """
    feature_bucket_splits_dict = dict()
    # fit splitter
    for col in config.CONTINUOUS_COLUMNS:
        sparser = QuantileDiscretizer(numBuckets=config.CONTINUOUS_COLUMNS_BUCKET_NUM,
                                      inputCol=col,
                                      outputCol=config.BUCKET_FEATURE_PREFIX + col,
                                      relativeError=0.01,
                                      handleInvalid="error")
        sparser_model = sparser.fit(train_data)
        feature_bucket_splits_dict[col] = sparser_model.getSplits()
    # 保存 splits 方便上线时一致
    utils.write_to_hdfs(
        sc, config.CONTINUOUS_COLUMNS_BUCKETS,
        json.dumps(feature_bucket_splits_dict),
        overwrite=True)
    print("[INFO] save continuous columns buckets {0}"
          .format(config.CONTINUOUS_COLUMNS_BUCKETS))

    # transform data
    for k, v in feature_bucket_splits_dict.items():
        input_col, output_col, splits = k, config.BUCKET_FEATURE_PREFIX + k, v

        bucket_model = feature.Bucketizer(
            inputCol=input_col,
            outputCol=output_col,
            splits=splits
        )
        train_data = bucket_model.setHandleInvalid("skip").transform(train_data)
        test_data = bucket_model.setHandleInvalid("skip").transform(test_data)

        train_data.select(output_col).show(10, False)

        print("[INFO] continuous sparse transform {0} to {1}".format(input_col, output_col))

    return train_data, test_data, feature_bucket_splits_dict


if __name__ == '__main__':
    spark_ctx = SparkContext()
    spark_sess = SparkSession \
        .builder \
        .appName("din feature engineering") \
        .enableHiveSupport() \
        .getOrCreate()

    train_dataset = _get_data(spark_sess, "", "").repartition(3000).persist()
    test_dataset = _get_data(spark_sess, "", "").repartition(3000).persist()
    process(spark_ctx, train_dataset, test_dataset)
    spark_ctx.stop()
    spark_sess.stop()
