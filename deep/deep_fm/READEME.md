# DeepFM
### 使用keras 实现 deepFM

Steps:<br>
* config <br>
配置样本查询sql，特征变量及其他配置项 <br>

* feature_engineering <br>
生成libsvm格式的TFRecord <br>
连续变量标准化/分桶，结果以libsvm形式存储到hdfs中 <br>

* deepfm_train <br>