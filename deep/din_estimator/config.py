#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: config.py
@time: 2019/11/4 上午11:06
@desc:
"""

INTEREST_MAX_LEN = 20

CONTINUOUS_COLUMNS_BUCKETS = ""

FEATURE_DICT = ""

VOCABULARY_DICT = ""

FINAL_FEATURE_LIST = ""

SAMPLE_SAVE_DIR = ""

SCALER_MODEL = ""

ONE_HOT_MODEL = ""

CONTINUOUS_COLUMNS_BUCKET_NUM = 20
BUCKET_FEATURE_PREFIX = "bucket_"
HASH_FEATURE_PREFIX = "hash_"
ONE_HOT_PREFIX = "one_hot_"

SAMPLE_SQL_TPL = """

"""

COLUMNS = [

]

CONTINUOUS_COLUMNS = [

]

CATEGORICAL_COLUMNS = [

]

HIGH_DIMENSIONAL_SPARSE_COLUMNS = {
    "passenger_id": 1000000
}

ITEM_ID_COLUMNS = [

]

INTEREST_COLUMNS = [

]

USE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + ['label']