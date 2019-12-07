#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: feature_engineering.py
@time: 2019/11/17 下午7:02
@desc:
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StandardScaler, StandardScalerModel

import deep.wide_deep.config as config
from deep import utils

parser = argparse.ArgumentParser()
parser.add_argument('--is_train', dest='is_train', action='store_true')
parser.add_argument('--is_test', dest='is_train', action='store_false')
parser.add_argument('--start', dest='start')
parser.add_argument('--end', dest='end')
parser.set_defaults(is_train=True)
parser.set_defaults(start='20191004')
parser.set_defaults(end='20191017')


def get_data(spark: SparkSession, args):
    sql = config.SAMPLE_SQL_TPL.format(
        START_DATE=args.start,
        END_DATE=args.end
    )
    print(sql)
    return spark.sql(sql).na.fill(-999.0)


def generate_poi_vocabulary(spark: SparkSession, args):
    sql = config.POI_VOCABULARY_SQL_TPL.format(
        START_DATE=args.start,
        END_DATE=args.end
    )
    print(sql)
    save_path = config.SAMPLE_SAVE_DIR + "/poi_vocabulary/"
    spark.sql(sql).repartition(1).write.csv(save_path, mode="overwrite", sep="\t", header="true")


def normalize_continuous_features(scale_feature_names: list, df: DataFrame, args):
    orignail_cols = df.columns

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

    df = df.select(orignail_cols
                   + [F.col("scaled_features")[i] for i in range(len(scale_feature_names))])

    for idx, feature in enumerate(scale_feature_names):
        df = df.withColumnRenamed(feature, feature + "_original") \
            .withColumnRenamed("scaled_features[{0}]".format(str(idx)), feature)

    if args.is_train:
        save_path = config.SAMPLE_SAVE_DIR + "/train/"
    else:
        save_path = config.SAMPLE_SAVE_DIR + "/test/"

    df.repartition(1000).write.csv(save_path, mode="overwrite", sep="\t", header="true")


if __name__ == '__main__':
    sparkSess = SparkSession \
        .builder \
        .appName("wdl feature engineering") \
        .enableHiveSupport() \
        .getOrCreate()
    args = parser.parse_args(sys.argv[1:])
    # 0. get data
    data = get_data(sparkSess, args)
    # 1. feature scale
    scaled_features = config.CONTINUOUS_COLUMNS
    normalize_continuous_features(sparkSess, scaled_features, data, args)
    # 2. poi vocabulary
    if args.is_train:
        generate_poi_vocabulary(sparkSess, args)
