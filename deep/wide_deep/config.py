#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: config.py
@time: 2019/11/17 下午6:54
@desc:
"""
# scaler path
SCALER_MODEL = ""

# sample path
SAMPLE_SAVE_DIR = ""

# 获取样本 hive sql
POI_VOCABULARY_SQL_TPL = """

"""

# 获取样本 hive sql
SAMPLE_SQL_TPL = """

"""

# 全量特征
COLUMNS = [

]

# 连续变量
CONTINUOUS_COLUMNS = [

]

# 离散变量
CATEGORICAL_COLUMNS = [

]

USE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + ['label']
