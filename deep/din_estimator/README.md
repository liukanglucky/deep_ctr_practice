# Deep Interest Network
### 使用 tensorflow.estimator 实现 DIN (增加了wide部分)

Step:<br>
* 数据准备
> What features can feed in ?
>> bucket continuous features
>> sparse feature
>> candidate item ids (goods_id, categories_id ... )
>> user interest history ids sequence (goods_id, categories_id ... )
>>> <font color="red">！！！TIPS: 需要预先将interest sequence padding成等长，便于矩阵运算。网络中会mask掉padding的0</font>
* 数据样例
>> features_value: [1, 1, 2 ...] <br>
>> numeric_features: [0.3, 0.2, 0.6, 0.85 ...] <br>
>> candidate item goods_id: 1006019 <br>
>> candidate item cate_id:  1007 <br>
>> user interest goods_id seq: [0, 0, 0, 0, 1006019] <br>
>> user interest cate_id seq: [0, 0, 0, 0, 10007] <br>

* din_train <br>