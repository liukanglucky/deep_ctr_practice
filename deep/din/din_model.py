#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: din_model.py
@time: 2019/12/6 下午8:01
@desc:
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def attention(x):
    """
    attention layer
    Tips: 无法定义成 keras.Layers save_model会报错
    :param x:
    :return:
    """
    assert len(x) == 3
    attention_hidden_units = (80, 40, 1)
    attention_activation = "sigmoid"
    i_emb, hist_emb, hist_len = x[0], x[1], x[2]
    hidden_units = K.int_shape(hist_emb)[-1]
    max_len = tf.shape(hist_emb)[1]

    i_emb = tf.tile(i_emb, [1, max_len])  # (batch_size, max_len * hidden_units)
    i_emb = tf.reshape(i_emb,
                       [-1, max_len, hidden_units])  # (batch_size, max_len, hidden_units)
    concat = K.concatenate([i_emb, hist_emb, i_emb - hist_emb, i_emb * hist_emb],
                           axis=2)  # (batch_size, max_len, hidden_units * 4)

    for i in range(len(attention_hidden_units)):
        activation = None if i == 2 else attention_activation
        outputs = keras.layers.Dense(attention_hidden_units[i], activation=activation,
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                                         scale=1.0, mode="fan_avg", distribution="normal"),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     name="attention_fc_{0}".format(i))(concat)
        concat = outputs

    outputs = tf.reshape(outputs, [-1, 1, max_len])  # (batch_size, 1, max_len)

    mask = tf.sequence_mask(hist_len, max_len)  # (batch_size, 1, max_len)
    padding = tf.ones_like(outputs) * (-1e12)
    outputs = tf.where(mask, outputs, padding)

    # 对outputs进行scale
    outputs = outputs / (hidden_units ** 0.5)
    outputs = K.softmax(outputs)

    outputs = tf.matmul(outputs, hist_emb)  # batch_size, 1, hidden_units)

    outputs = tf.squeeze(outputs)  # (batch_size, hidden_units)

    return outputs


def din(item_count, cate_count,
        deep_field_size,
        deep_feature_size,
        continuous_feature_size,
        emb_size,
        hidden_units=32,
        learning_rate=1,
        learning_rate_decay=0.01,
        drop_rate=0.2,
        deep_layers=(80, 40, 1)):
    """
    DIN  可接收稠密特征、稀疏特征、item ids、user interest ids sequence
    :param item_count: 商品数
    :param cate_count: 类别数
    :param hidden_units: 隐藏单元数
    :param deep_field_size
    :param deep_feature_size
    :param continuous_feature_size
    :param emb_size
    :param learning_rate
    :param learning_rate_decay
    :param drop_rate
    :param deep_layers
    :return: model
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="normal")

    # !!! Input start !!!
    # 候选item
    target_item = keras.layers.Input(shape=(1,), name='target_item', dtype="int32")
    # 候选item对应的所属类别
    target_cate = keras.layers.Input(shape=(1,), name='target_cate', dtype="int32")
    # user hist
    hist_item_seq = keras.layers.Input(shape=(None,), name="hist_item_seq", dtype="int32")
    # user hist cate
    hist_cate_seq = keras.layers.Input(shape=(None,), name="hist_cate_seq", dtype="int32")
    # hist length
    hist_len = keras.layers.Input(shape=(1,), name='hist_len', dtype="int32")
    # deep feature index
    deep_feat_index = keras.layers.Input(shape=(deep_field_size,), name="deep_feat_index")
    # deep feature value
    deep_feat_value = keras.layers.Input(shape=(deep_field_size,), name="deep_feat_value")
    # continuous_feature
    continuous_feature_value = keras.layers.Input(shape=(continuous_feature_size,),
                                                  name="continuous_feature_value")
    # !!! Input end !!!

    # !!! deep part start !!!
    # libsvm 特征处理
    embeddings = keras.layers.Embedding(deep_feature_size, emb_size,
                                        name='deep_feature_embedding',
                                        embeddings_initializer=tf.keras.initializers.VarianceScaling(
                                            scale=1.0, mode="fan_avg", distribution="normal"),
                                        embeddings_regularizer=tf.keras.regularizers.l2(0.01)
                                        )(deep_feat_index)
    feat_value = keras.layers.Reshape((deep_field_size, 1),
                                      name="deep_feat_value_reshape")(deep_feat_value)
    embeddings = keras.layers.Multiply(name="deep_feature_embedding_multiply")(
        [embeddings, feat_value])

    deep_emb_dense = keras.layers.Reshape((deep_field_size * emb_size,),
                                          name="deep_emb_dense_reshape")(embeddings)

    # concat continuous feature & sparse features
    deep_dense = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [deep_emb_dense, continuous_feature_value])
    deep_dense = keras.layers.Dense(hidden_units, activation="sigmoid",
                                    kernel_initializer=initializer,
                                    name="deep_dense")(deep_dense)
    deep_dense = keras.layers.BatchNormalization()(deep_dense)
    deep_dense = keras.layers.Dropout(rate=drop_rate)(deep_dense)
    # !!! deep part end !!!

    # !!! attention part start !!!
    item_emb = keras.layers.Embedding(input_dim=item_count,
                                      output_dim=hidden_units // 2,
                                      name="item_emb",
                                      embeddings_initializer=tf.keras.initializers.VarianceScaling(
                                          scale=1.0, mode="fan_avg", distribution="normal"),
                                      embeddings_regularizer=tf.keras.regularizers.l2(0.01)
                                      )
    cate_emb = keras.layers.Embedding(input_dim=cate_count,
                                      output_dim=hidden_units // 2,
                                      name="cate_emb",
                                      embeddings_initializer=tf.keras.initializers.VarianceScaling(
                                          scale=1.0, mode="fan_avg", distribution="normal"),
                                      embeddings_regularizer=tf.keras.regularizers.l2(0.01)
                                      )
    item_b = keras.layers.Embedding(input_dim=item_count, output_dim=1,
                                    name="item_bias",
                                    embeddings_initializer=keras.initializers.Constant(0.0),
                                    embeddings_regularizer=tf.keras.regularizers.l2(0.01)
                                    )

    # get target bias embedding
    target_item_bias_emb = item_b(target_item)
    target_item_bias_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(target_item_bias_emb)

    # get target embedding
    target_item_emb = item_emb(target_item)
    target_cate_emb = cate_emb(target_cate)
    i_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [target_item_emb, target_cate_emb])
    i_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(i_emb)

    # get history item embedding
    hist_item_emb = item_emb(hist_item_seq)
    hist_cate_emb = cate_emb(hist_cate_seq)
    hist_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [hist_item_emb, hist_cate_emb])

    # 构建点击序列与候选的attention关系
    din_attention = attention([i_emb, hist_emb, hist_len])
    din_attention = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, hidden_units]))(din_attention)

    # !!! attention part end !!!

    # !!! concat feature dense start !!!
    din_item = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=-1))(
        [i_emb, din_attention, deep_dense])
    for i in range(0, len(deep_layers)):
        activation = None if i == len(deep_layers) - 1 else "sigmoid"
        din_item = keras.layers.Dense(deep_layers[i],
                                      kernel_initializer=initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      activation=activation,
                                      name="concat_dense_{0}".format(i)
                                      )(din_item)
        din_item = keras.layers.BatchNormalization()(din_item)
        din_item = keras.layers.Dropout(rate=drop_rate)(din_item)
    # !!! concat feature dense end !!!

    logits = keras.layers.Add()([din_item, target_item_bias_emb])
    output = keras.layers.Activation('sigmoid')(logits)

    model = keras.models.Model(
        inputs=[hist_item_seq, hist_cate_seq, target_item, target_cate, hist_len,
                deep_feat_index, deep_feat_value, continuous_feature_value],
        outputs=output, name="model")

    model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=learning_rate, decay=learning_rate_decay),
        loss="binary_crossentropy")

    return model
