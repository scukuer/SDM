# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 25 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)


class ConcatAttention(tf.keras.layers.Layer):
    """
    :param query: [batch_size, T, C_q]
    :param key:   [batch_size, T, C_k]
    :return:      [batch_size, 1, T]
    query_size should keep the same dim with key_size
    """

    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        self.projection_layer = tf.keras.layers.Dense(units=1, activation='tanh')
        super(ConcatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ConcatAttention` layer should be called on a list of 2 tensors')
        super(ConcatAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # query: (batch_size, T, dim_q)
        # key:   (batch_size, T, dim_k)
        query, key = inputs
        q_k = tf.concat([query, key], axis=-1)
        output = self.projection_layer(q_k)
        if self.scale:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        output = tf.transpose(output, [0, 2, 1])   # [batch_size, 1, T]
        return output

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[1][1]

    def compute_mask(self, inputs, mask):
        return mask
