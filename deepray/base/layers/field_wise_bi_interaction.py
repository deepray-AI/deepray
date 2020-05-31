#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================

"""
Field-Wise BiInteraction
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class FieldWiseBiInteraction(tf.keras.layers.Layer):
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
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        super(FieldWiseBiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_fields = len(input_shape)
        self.embedding_size = input_shape[0][1]

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
