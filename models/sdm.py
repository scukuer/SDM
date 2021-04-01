# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 25 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dense

src_path = os.path.abspath("..")
sys.path.append(src_path)

from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.initializers import Zeros
from layers.attention_sequence_pooling_layer import AttentionSequencePoolingLayer
from layers.self_multi_head_attention import SelfMultiHeadAttention
from layers.dynamic_multi_rnn import DynamicMultiRNN
from utils.common_func import concat_func
from layers.user_attention import UserAttention
from utils.embedding_index import EmbeddingIndex
from utils.conf import *


class SeqDeepMatch(tf.keras.Model):
    """
    SDM模型
    """
    def __init__(self, field_str, user_id_size, item_id_size, channel_id_size, province_id_size,
                 user_id_dim, item_id_dim, channel_id_dim, province_id_dim,
                 num_samples=200, seq_mask_zero=True, l2_reg_embedding=1e-6):

        super(SeqDeepMatch, self).__init__()

        self.num_sampled = num_samples
        self.item_voc_size = item_id_size
        self.zero_bias = self.add_weight(shape=[item_id_size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")

        self.field_list = field_str.split(",")    # 字段名称集合

        self.user_id_embedding = Embedding(input_dim=user_id_size, output_dim=user_id_dim,
                                           mask_zero=seq_mask_zero, embeddings_regularizer=l2(l2_reg_embedding))
        self.item_id_embedding = Embedding(input_dim=item_id_size, output_dim=item_id_dim,
                                           mask_zero=seq_mask_zero, embeddings_regularizer=l2(l2_reg_embedding))
        self.channel_id_embedding = Embedding(input_dim=channel_id_size, output_dim=channel_id_dim,
                                              mask_zero=seq_mask_zero, embeddings_regularizer=l2(l2_reg_embedding))
        self.province_id_embedding = Embedding(input_dim=province_id_size, output_dim=province_id_dim,
                                               mask_zero=seq_mask_zero, embeddings_regularizer=l2(l2_reg_embedding))

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:  必须传入字典格式 && self.field_list保持一致
        :param training:
        :param mask:
        :return:
        """
        if not isinstance(inputs, dict):
            raise ValueError('A `sdm` model inputs type should be dict.')

        # if len(inputs) != len(self.field_list):
        #     raise ValueError('A `sdm` model inputs size should be equal.')

        input_keys = inputs.keys

        for key in input_keys:
            if self.field_list.contains(key):
                continue
            else:
                raise ValueError('A `sdm` model inputs key_name is not right.')

        user_emb = self.user_id_embedding(inputs["user_id"])              # (batch_size, 1, user_id_dim)
        item_emb = self.item_id_embedding(inputs["item_id"])              # (batch_size, 1, item_id_dim)
        province_emb = self.province_id_embedding(inputs["province_id"])  # (batch_size, 1, province_id_dim)

        short_seq_len = inputs["short_seq_len"]    # (batch_size, 1)
        long_seq_len = inputs["long_seq_len"]      # (batch_size, 1)

        # (batch_size, short_seq_len, item_id_dim)
        short_seq_item_id_emb = self.item_id_embedding(inputs["short_seq_item_id"])
        # (batch_size, long_seq_len, item_id_dim)
        long_seq_item_id_emb = self.item_id_embedding(inputs["long_seq_item_id"])

        # (batch_size, short_seq_len, item_id_dim)
        short_seq_channel_id_emb = self.channel_id_embedding(inputs["short_seq_channel_id"])
        # (batch_size, long_seq_len, item_id_dim)
        long_seq_channel_id_emb = self.channel_id_embedding(inputs["long_seq_channel_id"])

        # 1、user vector
        user_emb_list = [user_emb, province_emb]

        user_emb = concat_func(user_emb_list)
        # (batch_size, 1, units)
        user_emb_output = Dense(units, activation=dnn_activation, name="user_emb_output")(user_emb)

        # 2、long part
        prefer_emb_list = [long_seq_item_id_emb, long_seq_channel_id_emb]

        prefer_att_outputs = []

        for i, prefer_emb in enumerate(prefer_emb_list):
            # 长期行为中side info和user profile做attention：有mask
            prefer_attention_output = AttentionSequencePoolingLayer(dropout_rate=0)(
                [user_emb_output, prefer_emb, long_seq_len])

            prefer_att_outputs.append(prefer_attention_output)
        prefer_att_concat = concat_func(prefer_att_outputs)

        # (batch_size, 1, units)
        prefer_output = Dense(units, activation=dnn_activation, name="prefer_output")(prefer_att_concat)

        # 3、short part
        short_emb_list = [short_seq_item_id_emb, short_seq_channel_id_emb]
        short_emb_concat = concat_func(short_emb_list)

        # (batch_size, short_seq_len, units)
        short_emb_input = Dense(units, activation=dnn_activation, name="short_emb_input")(short_emb_concat)

        # (batch_size, short_seq_len, units)
        short_rnn_output = DynamicMultiRNN(num_units=units, return_sequence=True, num_layers=rnn_layers,
                                           num_residual_layers=rnn_num_res,
                                           dropout_rate=dropout_rate)([short_emb_input, short_seq_len])

        short_att_output = SelfMultiHeadAttention(num_units=units, head_num=num_head)([short_rnn_output, short_seq_len])

        # (batch_size, 1, units)
        short_output = UserAttention(num_units=units, activation=dnn_activation, use_res=True,
                                     dropout_rate=dropout_rate)([user_emb_output, short_att_output, short_seq_len])

        gate_input = concat_func([prefer_output, short_output, user_emb_output])
        gate = Dense(units, activation='sigmoid')(gate_input)  # (batch_size, 1, units)

        gate_output = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1 - x[0], x[2]))(
            [gate, short_output, prefer_output])  # (batch_size, 1, units)
        gate_output_reshape = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(gate_output)    # (batch_size, units)

        # prepare for sampled_softmax_loss
        item_index = EmbeddingIndex(list(range(self.item_voc_size)))(0)  # [vocab_size,]

        item_emb = self.item_id_embedding(item_index)  # [vocab_size, dim]

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=item_emb,                # 全体的item embedding
                                                         biases=self.zero_bias,
                                                         labels=inputs["item_id"],        # 传入的目标 batch item id
                                                         inputs=gate_output_reshape,      # 模型的中间输入
                                                         num_sampled=self.num_sampled,
                                                         num_classes=self.item_voc_size,  # item的词典大小
                                                         ))

        return float(loss)












