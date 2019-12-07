#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: generate_libsvm.py
@time: 2019/11/17 下午7:09
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
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StandardScaler, StandardScalerModel

from deep.deep_fm import config
from deep import utils

parser = argparse.ArgumentParser()
parser.add_argument('--is_train', dest='is_train', action='store_true')
parser.add_argument('--is_test', dest='is_train', action='store_false')
parser.add_argument('--start', dest='start')
parser.add_argument('--end', dest='end')
parser.set_defaults(is_train=True)
parser.set_defaults(start='')
parser.set_defaults(end='')

UNKNOWN_VALUE_KEY = -666.0


def process(spark: SparkSession, sc: SparkContext, args, continuous_sparse=True):
    data = _get_data(spark, args).cache()
    print("[INFO] original features:")
    data.show(2, False)

    final_sparse_features = list()
    final_continuous_features = list()
    bucket_feature_prefix = "bucket_"

    # 0. 数值变量处理
    if continuous_sparse:
        # 连续变量离散化
        data, continuous_feature_bucket_splits_dict = \
            _sparse_feature(data, args, bucket_feature_prefix, spark, sc)
        final_sparse_features.extend(
            [bucket_feature_prefix + col for col in continuous_feature_bucket_splits_dict.keys()])
        data = data.cache()
        data.show(5, False)
        print("[INFO][SUCCEED] bucket continuous feature succeed!")
    else:
        # 连续变量标准化
        data, final_continuous_features = _normalize_feature(data, args)
        continuous_feature_bucket_splits_dict = dict()
        data = data.cache()
        data.show(5, False)
        print("[INFO][SUCCEED] normalize continuous feature succeed!")

    # 1. 生成离散变量词表
    final_sparse_features = final_sparse_features + config.CATEGORICAL_COLUMNS
    categorical_feature_vocabulary_dict_path = \
        _gen_cat_feature_vocabulary(
            data,
            config.CATEGORICAL_COLUMNS,
            continuous_feature_bucket_splits_dict,
            bucket_feature_prefix,
            args, spark, sc)
    print("[INFO][SUCCEED] generate categorical feature vocabularies succeed!")

    # 2. 获取feature_value_index_dict
    feature_value_index_dict = \
        _get_or_create_feature_value_index_dict(
            final_sparse_features, final_continuous_features,
            categorical_feature_vocabulary_dict_path, args, sc)
    print("[INFO][SUCCEED] generate feature value-index dict succeed!")

    # 3. 生成X_index, X_value, y 数据格式
    final_use_features, data = _transform_data(final_sparse_features, final_continuous_features,
                                               data, feature_value_index_dict, sc)
    print("[INFO][SUCCEED] transform data succeed!")

    # 4. save data
    utils.write_to_hdfs(sc, config.FINAL_FEATURE_LIST, ', '.join(final_use_features))
    if args.is_train:
        path = config.SAMPLE_SAVE_DIR + "/train/"
    else:
        path = config.SAMPLE_SAVE_DIR + "/test/"
    data.repartition(100).write.mode('overwrite').format("tfrecords").option(
        "recordType", "Example").save(path)
    print("[INFO][SUCCEED] save data succeed!")


def _transform_data(sparse_features, continuous_features,
                    data: DataFrame, feature_dict_path: str, sc: SparkContext):
    """
    转换数据， 生成deepFM所需格式的TFRecord
    数据格式形如: trace_id, feature_index, feature_values
    :param sparse_features:
    :param continuous_features:
    :param data:
    :param feature_dict_path:
    :param sc:
    :return:
    """

    def _get_feature_value_index_udf(broadcast_feature_dict, feature_name):
        feature_dict = broadcast_feature_dict.value

        def _get_feature_value_index_wrapper(feature_value):
            # 离散变量返回对应值的index
            if feature_name in sparse_features:
                if str(feature_value) in feature_dict[feature_name].keys():
                    return int(feature_dict[feature_name][str(feature_value)])
                else:
                    return int(feature_dict[feature_name][str(UNKNOWN_VALUE_KEY)])
            # 连续变量只有一个index
            else:
                return feature_dict[feature_name]

        return F.udf(lambda x: _get_feature_value_index_wrapper(x))

    broadcast_feature_dict = \
        sc.broadcast(json.loads(utils.read_from_hdfs(sc, feature_dict_path)))

    features = broadcast_feature_dict.value.keys()  # 由于dict无需，这里统一获取feature list

    for col in features:
        data = data \
            .withColumn("feature_index_" + col,
                        _get_feature_value_index_udf(
                            broadcast_feature_dict, col)(data[col]).cast("float"))
        if col in continuous_features:
            data = data.withColumn("feature_value_" + col, data[col].cast("float"))
        else:
            data = data.withColumn("feature_value_" + col, F.lit(1).cast("float"))

    data = data.cache()
    print("[INFO] transformed features: ")
    data.show(5, False)
    # vectorAssembler
    feature_index_vector_assembler = feature.VectorAssembler(
        inputCols=["feature_index_" + f for f in features], outputCol="feature_index")
    feature_value_vector_assembler = feature.VectorAssembler(
        inputCols=["feature_value_" + f for f in features], outputCol="feature_value")
    data = feature_index_vector_assembler.transform(data)
    data = feature_value_vector_assembler.transform(data)
    data = data.select("trace_id", "feature_index", "feature_value", "label")

    return features, data


def _get_data(spark: SparkSession, args):
    sql = config.SAMPLE_SQL_TPL.format(
        cols=','.join(col for col in config.COLUMNS),
        START_DATE=args.start,
        END_DATE=args.end)
    print("[INFO] get data sql: {0}".format(sql))
    return spark.sql(sql).na.fill(-999.0)


def _gen_cat_feature_vocabulary(
        data: DataFrame,
        orig_sparse_features: list,
        transform_sparse_features_dict: dict,
        bucket_feature_prefix,
        args, spark: SparkSession, sc: SparkContext):
    """
    离散变量生成词表
    :param data:
    :param orig_sparse_features:
    :param transform_sparse_features_dict
    :param bucket_feature_prefix
    :param args:
    :param sc:
    :return:
    """
    tmp_table = "tmp_table"
    data.createOrReplaceTempView(tmp_table)
    if args.is_train:
        cat_feature_vocabulary_dict = dict()
        for col in orig_sparse_features:
            vocabulary_df = spark.sql(
                "select distinct {col} as {col} from {table}".format(
                    col=col, table=tmp_table)).toPandas()
            cat_feature_vocabulary_dict[col] = vocabulary_df[col].tolist()

        for k, v in transform_sparse_features_dict.items():
            cat_feature_vocabulary_dict[bucket_feature_prefix + k] = [i for i in range(0, len(v))]

        utils.write_to_hdfs(
            sc, config.VOCABULARY_DICT, json.dumps(cat_feature_vocabulary_dict), overwrite=True)
        print("[INFO] VOCABULARY_DICT write success: {0}".format(config.VOCABULARY_DICT))
    return config.VOCABULARY_DICT


def _get_or_create_feature_value_index_dict(
        sparse_features, continuous_features,
        categorical_feature_vocabulary_dict_path: str,
        args,
        sc: SparkContext):
    """
    获取feature_value到feature_index映射的 dict path
    :param sparse_features:
    :param continuous_features:
    :param categorical_feature_vocabulary_dict_path:
    :param args:
    :param sc:
    :return:
    """
    if args.is_train:
        categorical_feature_vocabulary_dict = json.loads(
            utils.read_from_hdfs(sc, categorical_feature_vocabulary_dict_path))
        feature_dict = \
            _generate_feature_value_index_dict(
                sparse_features, continuous_features,
                categorical_feature_vocabulary_dict
            )
        utils.write_to_hdfs(sc, config.FEATURE_DICT, json.dumps(feature_dict))
        print("[INFO] feature dict save success: {0}".format(config.FEATURE_DICT))

    return config.FEATURE_DICT


def _generate_feature_value_index_dict(
        sparse_features, continuous_features,
        categorical_feature_vocabulary_dict: dict):
    """
    构建feature_value到feature_index映射的 dict
    feature_dict: {
        # 离散变量
        col1:{
            col1_v1: index,
            col1_v2: index,
            ...
        },
        # 连续变量
        col2: index,
        ...
    }
    :param sparse_features:
    :param continuous_features:
    :param categorical_feature_vocabulary_dict
    :return:
    """
    feature_dict = {}
    total_feature = 0

    for col in sparse_features:
        unique_val = categorical_feature_vocabulary_dict[col] + [UNKNOWN_VALUE_KEY]
        feature_dict[col] = dict(
            zip(unique_val, range(total_feature, len(unique_val) + total_feature)))
        total_feature += len(unique_val)
    for col in continuous_features:
        feature_dict[col] = total_feature
        total_feature += 1

    print("[INFO] total feature is: " + str(total_feature))
    return feature_dict


def _sparse_feature(data: DataFrame, args, bucket_feature_prefix,
                    spark: SparkSession, sc: SparkContext):
    """
    连续特征离散化
    :param data:
    :param args:
    :param bucket_feature_prefix:
    :param sc: sparkContext
    :return: 含有离散化特征的DataFrame, 分桶splits
    """
    if args.is_train:
        feature_bucket_splits_dict = _gen_bucketizer_splits(spark, data)
        utils.write_to_hdfs(
            sc, config.CONTINUOUS_COLUMNS_BUCKETS,
            json.dumps(feature_bucket_splits_dict),
            overwrite=True)
        print("[INFO] save continuous columns buckets {0}"
              .format(config.CONTINUOUS_COLUMNS_BUCKETS))
    else:
        feature_bucket_splits_dict = \
            json.loads(utils.read_from_hdfs(sc, config.CONTINUOUS_COLUMNS_BUCKETS))
        print("[INFO] load continuous columns buckets {0}"
              .format(config.CONTINUOUS_COLUMNS_BUCKETS))

    for k, v in feature_bucket_splits_dict.items():
        input_col, output_col, splits = k, bucket_feature_prefix + k, v
        print("[INFO] bucketizer transform {0} to {1}".format(input_col, output_col))
        bucketizer = feature.Bucketizer(
            inputCol=input_col,
            outputCol=output_col,
            splits=splits
        )
        data = bucketizer.setHandleInvalid("skip").transform(data)

    return data, feature_bucket_splits_dict


def _normalize_feature(df: DataFrame, args):
    original_cols = df.columns
    scale_feature_names = config.CONTINUOUS_COLUMNS

    df = df.withColumn('dense_features',
                       utils.udf_gen_features_v2()(F.array(*scale_feature_names)))

    df = df.withColumn('dense_vector',
                       utils.array_to_vector_udf(F.col('dense_features')))

    scaler_file = config.SCALER_MODEL
    if args.is_train:
        scaler = StandardScaler(inputCol="dense_vector",
                                outputCol="scaled_vector",
                                withMean=False, withStd=True)
        scaler_model = scaler.fit(df)
        scaler_model.write().overwrite().save(scaler_file)
    else:
        scaler_model = StandardScalerModel.load(scaler_file)

    df = scaler_model.transform(df)

    df = df.withColumn('scaled_features',
                       utils.vector_to_array_udf(FloatType())(F.col('scaled_vector')))

    df = df.select(original_cols
                   + [F.col("scaled_features")[i] for i in range(len(scale_feature_names))])

    for idx, feature in enumerate(scale_feature_names):
        df = df.withColumnRenamed(feature, feature + "_original") \
            .withColumnRenamed("scaled_features[{0}]".format(str(idx)), feature)

    return df, scale_feature_names


def _gen_bucketizer_splits(spark, data: DataFrame):
    tmp_table_name = "data"

    def _gen_get_range_sql(columns):
        """
        获取除空值外的最大最小值
        :param columns:
        :return:
        """
        tmp_sql = ""
        for col in columns:
            tmp_sql += "min(if({col} = -999, NULL, {col})) as {col}_min, max({col}) as {col}_max," \
                .format(col=col)
        return "select {sql} from {table_name}".format(
            sql=tmp_sql[:-1],  # 去掉最后多余的逗号
            table_name=tmp_table_name
        )

    data.createOrReplaceTempView(tmp_table_name)
    sql = _gen_get_range_sql(config.CONTINUOUS_COLUMNS)
    print("[INFO] stat features range: " + sql)
    data_features_range = spark.sql(sql).cache()
    print("[INFO] features range: ")
    data_features_range.show(5, False)
    data_features_range = data_features_range.toPandas()
    data_features_range[data_features_range.columns] = \
        data_features_range[data_features_range.columns].apply(pd.to_numeric)

    feature_bucket_splits_dict = dict()

    # 只对CONTINUOUS_COLUMNS分桶
    for col in config.CONTINUOUS_COLUMNS:
        min = data_features_range[col + "_min"].loc[0]
        max = data_features_range[col + "_max"].loc[0]

        # 单值类特征 舍弃
        if min == max:
            print("[INFO] feature {0} only has 1 unique value, abandon.".format(col))
            # feature_bucket_splits_dict[col] = [-999] + [min]
        # 概率类特征默认10个分桶
        elif min >= 0 and max <= 1:
            feature_bucket_splits_dict[col] = \
                [-999] + list(np.linspace(min, max, 11)) + [float('inf')]
        # 数值型特征默认分50个桶
        else:
            feature_bucket_splits_dict[col] = \
                [-999] + list(np.linspace(min, max, 51)) + [float('inf')]
    print("[INFO] feature_bucket_splits_dict: " + str(feature_bucket_splits_dict))

    return feature_bucket_splits_dict


if __name__ == '__main__':
    spark_ctx = SparkContext()
    spark_sess = SparkSession \
        .builder \
        .appName("deepFM feature engineering") \
        .enableHiveSupport() \
        .getOrCreate()
    params = parser.parse_args(sys.argv[1:])
    process(spark_sess, spark_ctx, params, continuous_sparse=False)
    spark_ctx.stop()
    spark_sess.stop()
