#!/usr/bin/env python
# encoding: utf-8
"""
@author: liukang
@file: din_model.py
@time: 2019/12/6 下午8:34
@desc:
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/')

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.estimator.canned import optimizers

from deep.din_estimator.input_fn import *


class DIN(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adam',
                 config=None,
                 warm_start_from=None,
                 wdl_only=False
                 ):

        self.optimizer = optimizers.get_optimizer_instance(optimizer, params["learning_rate"])

        def _model_fn(features, labels, mode, params):
            my_head = tf.estimator.BinaryClassHead(thresholds=[0.5])

            # deep feature
            deep_features = tf.compat.v1.feature_column.input_layer(
                features, params['deep_features'])

            # wide feature
            wide_features = tf.compat.v1.feature_column.input_layer(
                features, params['wide_features'])

            if wdl_only:
                deep_dense = tf.reshape(deep_features, shape=[-1, deep_features.shape[1]])
            else:
                # multi attention
                dense_features = [deep_features]
                # dense_features = [deep_features]
                for i in range(0, len(item_ids_features)):
                    item_id = features[item_ids_features[i]]
                    user_ids_seq = features[user_interest_ids_features[i]]
                    user_ids_len = features[user_interest_len[i]]

                    neg_user_ids_seq = features[user_neg_interest_ids_features[i]]
                    neg_user_ids_len = features[user_neg_interest_len[i]]

                    att_out = \
                        self.attention_layer(user_ids_seq, item_id, user_ids_len,
                                             item_ids_features[i])

                    neg_att_out = \
                        self.attention_layer(neg_user_ids_seq, item_id, neg_user_ids_len,
                                             item_ids_features[i], layer_prefix="neg_att_")

                    dense_features.append(att_out)
                    dense_features.append(neg_att_out)

                # 只是为了给层命名
                din_dense_shape = sum([4 * v if k in item_ids_features else 0
                                       for k, v in item_ids_features_emb_size.items()])
                deep_dense = tf.concat(list(dense_features), axis=1)
                deep_dense = tf.reshape(deep_dense,
                                        shape=[-1,
                                               deep_features.shape[1]
                                               + din_dense_shape],
                                        name="deep_dense")

            for i, units in enumerate(params['hidden_units']):
                deep_dense = keras.layers.Dense(units=units, activation=None)(deep_dense)
                deep_dense = keras.layers.BatchNormalization()(deep_dense)
                deep_dense = keras.layers.Activation("relu")(deep_dense)
                if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
                    deep_dense = keras.layers.Dropout(params['dropout_rate'])(deep_dense)

            # 只是为了给层命名
            deep_dense = tf.reshape(
                deep_dense,
                shape=[-1, deep_dense.shape[1]], name="deep_output")

            final_concat = tf.concat([
                deep_dense,
                wide_features
            ], axis=1, name="wide_deep_concat")

            logits = keras.layers.Dense(units=my_head.logits_dimension)(final_concat)

            return my_head.create_estimator_spec(
                features=features,
                mode=mode,
                labels=labels,
                logits=logits,
                train_op_fn=lambda loss:
                self.optimizer.minimize(loss,
                                        global_step=tf.compat.v1.train.get_global_step())
            )

        super(DIN, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config, params=params,
            warm_start_from=warm_start_from)

    def attention_layer(self, seq_ids, tid, seq_len, id_type, layer_prefix=""):
        with tf.compat.v1.variable_scope(layer_prefix + "attention_" + id_type):
            embedding_size = self._params["embedding_size"][id_type]
            embeddings = keras.layers.Embedding(input_dim=self._params["vocab_size"][id_type],
                                                output_dim=embedding_size)

            seq_emb = embeddings(seq_ids)  # shape(batch_size, max_seq_len, embedding_size)
            tid_emb = tf.squeeze(
                embeddings(tid),
                name=layer_prefix + "item_id_emb_" + id_type)  # shape(batch_size, embedding_size)
            max_seq_len = tf.shape(seq_ids)[1]  # padded_dim

            u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])
            a_emb = tf.reshape(
                tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, embedding_size])
            net = tf.concat([u_emb, u_emb - a_emb, a_emb, a_emb * u_emb], axis=1)
            for units in self._params['attention_hidden_units']:
                net = keras.layers.Dense(units=units, activation=None)(net)
                net = keras.layers.BatchNormalization()(net)
                net = keras.layers.Activation("relu")(net)

            att_wgt = keras.layers.Dense(units=1, activation=tf.sigmoid)(net)
            att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1],
                                 name=layer_prefix + "weight_" + id_type)
            wgt_emb = tf.multiply(seq_emb,
                                  att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
            # masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
            masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)
            att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1,
                                    name=layer_prefix + "attention_emb_" + id_type)

            att_out = tf.concat([att_emb, tid_emb], axis=1,
                                name=layer_prefix + id_type + "_att_output")

            return att_out
