# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 26 Mar, 2021
Author : chenlangscu@163.com
prefer from https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/layers/sequence.py
"""

import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

src_path = os.path.abspath("..")
sys.path.append(src_path)

import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class DynamicMultiRNN(Layer):
    def __init__(self, num_units=None, rnn_type='LSTM', return_sequence=True, num_layers=2, num_residual_layers=1,
                 dropout_rate=0.2, forget_bias=1.0, **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_residual_layers = num_residual_layers
        self.dropout = dropout_rate
        self.forget_bias = forget_bias
        super(DynamicMultiRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        if self.rnn_type == "LSTM":
            try:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
            except AttributeError:
                single_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
        elif self.rnn_type == "GRU":
            try:
                single_cell = tf.nn.rnn_cell.GRUCell(self.num_units, forget_bias=self.forget_bias)
            except AttributeError:
                single_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units, forget_bias=self.forget_bias)
        else:
            raise ValueError("Unknown unit type %s!" % self.rnn_type)

        # 0 = test, 1 = train
        dropout = self.dropout if tf.keras.backend.learning_phase() == 1 else 0
        try:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        except AttributeError:
            single_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        cell_list = []
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            if residual:
                try:
                    single_cell_residual = tf.nn.rnn_cell.ResidualWrapper(single_cell)  # 把输入concat到输出上一起返回
                except AttributeError:
                    single_cell_residual = tf.compat.v1.nn.rnn_cell.ResidualWrapper(single_cell)
                cell_list.append(single_cell_residual)
            else:
                cell_list.append(single_cell)
        if len(cell_list) == 1:
            self.final_cell = cell_list[0]
        else:
            try:
                self.final_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            except AttributeError:
                self.final_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)
        super(DynamicMultiRNN, self).build(input_shape)

    def call(self, input_list, mask=None, training=None):
        rnn_input, sequence_length = input_list

        try:
            with tf.name_scope("rnn"), tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                rnn_output, hidden_state = tf.nn.dynamic_rnn(self.final_cell, inputs=rnn_input,
                                                             sequence_length=tf.squeeze(sequence_length),
                                                             dtype=tf.float32, scope=self.name)
        except AttributeError:
            with tf.name_scope("rnn"), tf.compat.v1.variable_scope("rnn", reuse=tf.compat.v1.AUTO_REUSE):
                rnn_output, hidden_state = tf.compat.v1.nn.dynamic_rnn(self.final_cell, inputs=rnn_input,
                                                                       sequence_length=tf.squeeze(sequence_length),
                                                                       dtype=tf.float32, scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return None, 1, rnn_input_shape[2]

    def get_config(self, ):
        config = {'num_units': self.num_units, 'rnn_type': self.rnn_type, 'return_sequence': self.return_sequence,
                  'num_layers': self.num_layers,
                  'num_residual_layers': self.num_residual_layers, 'dropout_rate': self.dropout,
                  'forget_bias': self.forget_bias}
        base_config = super(DynamicMultiRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def debug():
    inputs = tf.random.normal((3, 2, 32))
    num_uints = inputs.get_shape().as_list()[-1]
    inputs_length = np.random.randint(low=0, high=2, size=(3, 1))
    print("inputs_length: ", inputs_length)
    rnn_output = DynamicMultiRNN(num_units=num_uints, return_sequence=True, num_layers=2,
                                 num_residual_layers=1, dropout_rate=0.2)([inputs, inputs_length])
    print("rnn_output shape: ", rnn_output.shape)   # (3, 2, 32)
    print("rnn_outout: ", rnn_output)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
