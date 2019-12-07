#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: config.py
@time: 2019/11/17 下午7:09
@desc:
"""

# 连续变量分桶 分割点保存路径
CONTINUOUS_COLUMNS_BUCKETS = ""

# feature index-value 映射dic保存路径
FEATURE_DICT = ""

# feature 词表字典保存路径
VOCABULARY_DICT = ""

FINAL_FEATURE_LIST = ""

SAMPLE_SAVE_DIR = ""

SCALER_MODEL = ""

SAMPLE_SQL_TPL = """

"""

COLUMNS = [

]

CONTINUOUS_COLUMNS = [
]

CATEGORICAL_COLUMNS = [

]

USE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + ['label']
