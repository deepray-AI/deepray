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
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


__all__ = [
    'FullyConnect',
    'FullyConnectv2',
]


class FullyConnect(Layer):
    """The Multilayer Perceptrons

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units[-1])`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units[-1])`.

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
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.units = [units] if isinstance(units, int) else units
        self.hidden_units = None
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.last_dim = None
        if 'input_shape' in kwargs:
            self.last_dim = kwargs['input_shape'][0]

        super(FullyConnect, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input_shape: {}".format(input_shape))
        if hasattr(self, 'last_dim') and self.last_dim is not None:
            self.hidden_units = [int(self.last_dim)] + self.units
            print("in build, hidden_units: {}".format(self.last_dim, self.hidden_units))
        else:
            self.hidden_units = [input_shape[-1]] + self.units

        print("hidden_units: {}".format(self.hidden_units))

        self.kernel = self.add_weight(name=self.name + '_kernel',
                                      shape=(self.hidden_units[0], self.hidden_units[1]),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        self.bias = self.add_weight(name=self.name + '_bias',
                                    shape=(self.units[0],),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        # Be sure to call this somewhere
        super(FullyConnect, self).build(input_shape)

    def call(self, inputs, trainable=None, **kwargs):
        mlp = inputs
        if isinstance(mlp, tf.sparse.SparseTensor):
            fc = tf.sparse.sparse_dense_matmul(mlp, self.kernel)
        else:
            if len(inputs.get_shape()) != 2:
                # Broadcasting is required for the inputs.
                fc = tf.tensordot(mlp, self.kernel, [[len(mlp.get_shape()) - 1], [0]])
            else:
                fc = tf.linalg.matmul(mlp, self.kernel)

        if self.use_bias:
            fc = tf.nn.bias_add(fc, self.bias)
        if self.activation is not None:
            fc = self.activation(fc)  # pylint: disable=not-callable

        mlp = fc

        return mlp

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(dims=input_shape)
        return input_shape
        # TODO: return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super(FullyConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FullyConnectv2(Layer):
    """The Multilayer Perceptrons

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units[-1])`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units[-1])`.

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
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.units = [units] if isinstance(units, int) else units
        self.hidden_units = None
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.last_dim = None
        if 'input_shape' in kwargs:
            self.last_dim = kwargs['input_shape'][0]

        super(FullyConnectv2, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input_shape: {}".format(input_shape))
        if hasattr(self, 'last_dim') and self.last_dim is not None:
            self.hidden_units = [int(self.last_dim)] + self.units
            print("in build, hidden_units: {}".format(self.last_dim, self.hidden_units))
        else:
            self.hidden_units = [input_shape[-1]] + self.units

        print("hidden_units: {}".format(self.hidden_units))

        self.kernel = self.add_weight(name=self.name + '_kernel',
                                      shape=(self.hidden_units[0], self.hidden_units[1]),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        self.bias = self.add_weight(name=self.name + '_bias',
                                    shape=(self.units[0],),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        # Be sure to call this somewhere
        super(FullyConnectv2, self).build(input_shape)

    def call(self, inputs, trainable=None, **kwargs):
        mlp = inputs
        if isinstance(mlp, tf.sparse.SparseTensor):
            fc = tf.sparse.sparse_dense_matmul(mlp, self.kernel)
        else:
            if len(inputs.get_shape()) != 2:
                # Broadcasting is required for the inputs.
                fc = tf.tensordot(mlp, self.kernel, [[len(input.get_shape()) - 1], [0]])
            else:
                fc = tf.linalg.matmul(mlp, self.kernel)

        square_sum_factor = tf.square(fc)
        sum_square_factor = tf.sparse.sparse_dense_matmul(tf.square(inputs),
                                                          tf.square(self.kernel))

        fm_embedding = tf.subtract(square_sum_factor, sum_square_factor)

        if self.use_bias:
            fc = tf.nn.bias_add(fc, self.bias)
        if self.activation is not None:
            fc = self.activation(fc)  # pylint: disable=not-callable

        mlp = fc

        return mlp, fm_embedding

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(dims=input_shape)
        return input_shape
        # TODO: return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super(FullyConnectv2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

