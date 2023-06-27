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
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer

__all__ = [
  'FieldWiseBiInteraction',
]


class FieldWiseBiInteraction(Layer):
  """Field-wise Bi-Interaction Layer used in FLEN, pooling the
     field-wise element-wise product of features into one single vector

    Input shape:
      2-D tensor with shape: `(batch_size, embedding_size * field_num)`.

    Output shape:
      2-D tensor with shape: `(batch_size, embedding_size)`.

    Arguments:
      num_fields: Positive integer, number of feature fields.
      embedding_size: Positive integer, each field's embedding size
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
               num_fields,
               embedding_size,
               use_bias=False,
               activation=None,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    self.num_fields = num_fields
    self.embedding_size = embedding_size
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)

    super(FieldWiseBiInteraction, self).__init__(**kwargs)

  def build(self, input_shape):
    if len(input_shape) != 2:
      raise ValueError("field_wise_bi_interaction input must be a 2D Tensor")

    self.kernel = self.add_weight(name='kernel',
                                  shape=(int(self.num_fields * (self.num_fields - 1) / 2), 1),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=(1,),
                                  initializer=self.bias_initializer,
                                  trainable=True)

    # Be sure to call this somewhere
    super(FieldWiseBiInteraction, self).build(input_shape)

  def call(self, inputs, trainable=None, **kwargs):
    left = []
    right = []
    for i in range(self.num_fields):
      for j in range(i + 1, self.num_fields):
        left.append(i)
        right.append(j)

    embeddings = tf.reshape(inputs, [-1, self.num_fields, self.embedding_size])
    embeddings_left = tf.gather(params=embeddings, indices=left, axis=1)
    embeddings_right = tf.gather(params=embeddings, indices=right, axis=1)
    embeddings_prod = tf.multiply(x=embeddings_left, y=embeddings_right)
    field_weighted_embedding = tf.multiply(x=embeddings_prod, y=self.kernel)
    field_weighted_embedding = tf.reduce_sum(field_weighted_embedding, axis=1)

    if self.use_bias:
      field_weighted_embedding = tf.nn.bias_add(field_weighted_embedding, self.bias)

    if self.activation is not None:
      field_weighted_embedding = self.activation(field_weighted_embedding)

    return field_weighted_embedding

  def compute_output_shape(self, input_shape):
    shape = self.embedding_size
    return tuple(shape)

  def get_config(self):
    config = {
      'num_fields': self.num_fields,
      'embedding_size': self.embedding_size,
      'use_bias': self.use_bias,
      'activation': self.activation,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
    }
    base_config = super(FieldWiseBiInteraction, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)
