# Deep Ctr Practice

[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.google.cn/api_docs/python/tf?hl=en "tensorflow2.0")
[![Python Version](https://img.shields.io/badge/Python-3.6+-blue.svg)]()
[![Pyspark Version](https://img.shields.io/badge/Pyspark-2.4.4+-green.svg)](http://spark.apache.org/docs/latest/api/python/pyspark.html "pyspark2.4.4")
[![Scala Version](https://img.shields.io/badge/Scala--red.svg)]()
[![Hadoop Version](https://img.shields.io/badge/Hadoop-2.7.2+-yellow.svg)]()

ctr预估深度模型 <br>
based on: tensorflow2.0

### deep_ctr_practice
* wide & deep <br>
实现： <br>
data type: csv <br>
load data: load csv from HDFS <br>
train: tf.estimatror.DNNLinearCombinedClassifier<br>  
* deepFM <br>
实现： <br>
data type: libsvm <br>
load data: load TFRecord from HDFS<br>
train: tf.keras <br> 

* DIN <br>
实现： <br>
data type: libsvm sparse features + numeric_features + candidate item goods_id + candidate item cate_id + user interest goods_id seq + user interest cate_id seq <br>
load data: load TFRecord from HDFS<br>
train: tf.keras <br>

* DIEN <br>
施工中<br> 

## Reference
http://aducode.github.io/posts/2016-08-02/write2hdfsinpyspark.html <br>
https://github.com/ymkigeg/keras-DeepFM <br>