# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 25 Mar, 2021
Author : chenlangscu@163.com
prefer from https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/layers/interaction.py
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.concat_attention import ConcatAttention
from layers.softmax_weight_sum import SoftmaxWeightedSum


class AttentionSequencePoolingLayer(tf.keras.layers.Layer):
    """
    :param query:           [batch_size, 1, C_q]
    :param keys:            [batch_size, T, C_k]
    :param keys_length:     [batch_size, 1]
    :return:                [batch_size, 1, C_k]
    """

    def __init__(self, dropout_rate=0, **kwargs):
        self.dropout_rate = dropout_rate
        self.concat_att = ConcatAttention()
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=False)
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SequenceFeatureMask` layer should be called '
                             'on a list of 3 inputs')

        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # queries:(?, 1, units)
        # keys:   (?, T, units)
        queries, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]  # T

        # tf.sequence_mask([[1],[2],[6]],5)
        # < tf.Tensor: shape = (3, 1, 5), dtype = bool, numpy =
        # array([[[True, False, False, False, False]],
        #        [[True, True, False, False, False]],
        #        [[True, True, True, True, True]]]) >
        key_masks = tf.sequence_mask(keys_length, hist_len)  # [batch_size,1, T]

        queries = tf.tile(queries, [1, hist_len, 1])  # [batch_size, T, units]
        attention_score = self.concat_att([queries, keys])  # [batch_size, 1, T]

        outputs = self.softmax_weight_sum([attention_score, keys, key_masks])
        # [batch_size, 1, units]
        return outputs

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[1][1]

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask
