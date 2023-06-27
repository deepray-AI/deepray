# -*- coding:utf-8 -*-
# Copyright 2019 The Jarvis Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

__all__ = [
    'BiInteraction',
]


class BiInteraction(Layer):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector

    Input shape:
      2-D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
      2-D tensor with shape: `(batch_size, input_dim)`.

    Arguments:
      units: list of positive integer, hidden size for each hidden layer.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      use_bn: bool. Whether use BatchNormalization before activation or not.
      keep_prob: float between 0 and 1. Fraction of the units to keep
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 use_bn=False,
                 keep_prob=1.0,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.units = [units] if isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_bn = use_bn
        self.keep_prob = keep_prob
        # NOTE(Ryan): To avoid "tf.function-decorated function tried to create variables on non-first call." problem
        # I created a shared BatchNormalization() here.
        if self.use_bn:
            self.bn = BatchNormalization()

        super(BiInteraction, self).__init__(**kwargs)

    def _bi_interaction(self,
                        inputs,
                        kernel=None,
                        bias=None,
                        activation=None,
                        use_bias=False):

        square_sum_factor = tf.square(tf.matmul(inputs, kernel))
        sum_square_factor = tf.matmul(tf.square(inputs), tf.square(kernel))
        bi_inter = tf.subtract(square_sum_factor, sum_square_factor)
        if use_bias:
            bi_inter = tf.nn.bias_add(bi_inter, bias)

        if activation is not None:
            bi_inter = activation(bi_inter)

        return bi_inter

    def build(self, input_shape):
        print("Tracing bi_interaction.build()")
        if len(input_shape) != 2:
            raise ValueError("bi_interaction input must be a 2D Tensor")

        last_dim = input_shape[-1]
        hidden_units = [int(last_dim)] + self.units
        layer_num = len(self.units)

        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True) for i in xrange(layer_num)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.units[i],),
                                     initializer=self.bias_initializer,
                                     trainable=True) for i in xrange(layer_num)]

        # Be sure to call this somewhere
        super(BiInteraction, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        print("Tracing bi_interaction.call()")
        bi_inter = inputs
        for i in range(len(self.units)):

            hidden = self._bi_interaction(inputs=bi_inter,
                                          kernel=self.kernels[i],
                                          bias=self.bias[i],
                                          activation=self.activation,
                                          use_bias=self.use_bias)
            if self.use_bn and training:
                hidden = self.bn(inputs=hidden)

            if training:
                hidden = Dropout(1 - self.keep_prob)(hidden)
            bi_inter = hidden

        return bi_inter

    def compute_output_shape(self, input_shape):
        if len(self.units) > 0:
            shape = input_shape[:-1] + (self.units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'use_bn': self.use_bn,
            'keep_prob': self.keep_prob
        }
        base_config = super(BiInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

