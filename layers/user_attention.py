# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 26 Mar, 2021
Author : chenlangscu@163.com
prefer from https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/layers/interaction.py
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.dot_attention import DotAttention
from layers.softmax_weight_sum import SoftmaxWeightedSum


class UserAttention(tf.keras.layers.Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param keys: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, 1, C]
    """

    def __init__(self, num_units=None, activation='tanh', use_res=True, dropout_rate=0, scale=True, seed=2020,
                 **kwargs):
        self.scale = scale
        self.num_units = num_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.use_res = use_res
        super(UserAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `UserAttention` layer should be called '
                             'on a list of 3 tensors')
        if self.num_units == None:
            self.num_units = input_shape[0][-1]
        self.dense = tf.keras.layers.Dense(self.num_units, activation=self.activation)
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, seed=self.seed)
        super(UserAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # user_query: (?, 1, 64)    keys: (?, 5, 64)   keys_length: (?, 1)
        user_query, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]

        # tf.sequence_mask([[1],[2],[6]],5)
        # < tf.Tensor: shape = (3, 1, 5), dtype = bool, numpy =
        # array([[[True, False, False, False, False]],
        #        [[True, True, False, False, False]],
        #        [[True, True, True, True, True]]]) >
        key_masks = tf.sequence_mask(keys_length, hist_len)
        query = self.dense(user_query)

        align = self.attention([query, keys])

        output = self.softmax_weight_sum([align, keys, key_masks])

        if self.use_res:
            output += keys
        return tf.reduce_mean(output, 1, keep_dims=True)

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[1][2]

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'num_units': self.num_units, 'activation': self.activation, 'use_res': self.use_res,
                  'dropout_rate': self.dropout_rate,
                  'scale': self.scale, 'seed': self.seed, }
        base_config = super(UserAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
