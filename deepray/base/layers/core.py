#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hailin Fu
#
# Distributed under terms of the GNU license.

"""
Base layers
"""
import tensorflow as tf


class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, is_training=None):
        if is_training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class DeepBlock(tf.keras.layers.Layer):
    def __init__(self, hidden, activation, prefix, sparse=False, use_bn=False, res_deep=False):
        super(DeepBlock, self).__init__()
        self.units = hidden
        self.activation = activation
        self.prefix = prefix
        self.sparse = sparse
        self.use_bn = use_bn
        self.res_deep = res_deep

    def build(self, input_shape):
        self.w = self.add_weight(name='{}_weight'.format(self.prefix),
                                 shape=[input_shape[-1], self.units],
                                 initializer='random_normal' if self.sparse else None,
                                 regularizer=tf.keras.regularizers.l1_l2(l1=self.flags.l1,
                                                                         l2=self.flags.l2) if self.sparse else None,
                                 trainable=True)
        self.b = self.add_weight(name='{}_bias'.format(self.prefix),
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()

    def call(self, inputs, is_training=None):
        v = tf.matmul(inputs, self.w) + self.b
        if self.use_bn:
            v = self.bn_layers(v, training=is_training)
        outputs = tf.keras.layers.Activation(self.activation)(v)
        if self.res_deep:
            return outputs + inputs
        return outputs


class DeepNet(tf.keras.layers.Layer):
    def __init__(self, hidden, activation, sparse=False, droprate=0.1, flags=None):
        super(DeepNet, self).__init__()
        self.hidden = hidden
        self.activation = activation
        self.sparse = sparse
        self.droprate = droprate
        self.flags = flags

    def build(self, input_shape):
        self.kernel = [DeepBlock(hs, self.activation, 'deep_{}'.format(i),
                                 self.sparse, self.flags) for i, hs in enumerate(self.hidden)]

    def call(self, x, is_training=None):
        for i, hs in enumerate(self.hidden):
            '''
            randomly sets elements to zero to prevent overfitting.
            '''
            x = CustomDropout(rate=self.droprate)(x, is_training=is_training)
            x = self.kernel[i](x, is_training=is_training)
        return x


class Linear(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='linear_w',
                                 shape=[input_shape[-1], self.units],
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='linear_b',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config


class SelfAttentiveDeepNet(tf.keras.layers.Layer):
    def __init__(self, hidden, activation, concat_last_deep, sparse=False):
        super().__init__()
        self.hidden = hidden
        self.activation = activation
        self.sparse = sparse
        self.concat_last_deep = concat_last_deep

    def call(self, raw_inputs, **kwargs):
        outputs = []
        with tf.name_scope('deep'):
            output = tf.concat(raw_inputs, -1)
            for i, hs in enumerate(self.hidden):
                output = CustomDropout(0.1)(output)
                output = DeepBlock(hidden=hs,
                                   activation=self.activation,
                                   prefix='deep_{}'.format(i),
                                   sparse=self.sparse)(output)
                outputs.append(output)
            """
            H: column wise matrix of each deep layer
            """
            H = tf.stack(outputs, axis=2)
            """
            S = H' * H
            """
            S = tf.matmul(tf.transpose(H, perm=[0, 2, 1]), H)
            """
            Column wise softmax as attention
            """
            attention = tf.nn.softmax(S, axis=1)
            """
            G = H * A
            """
            G = tf.matmul(H, attention)
            """
            Sum over deep layers
            """
            G = tf.reduce_sum(G, axis=-1)

            if self.concat_last_deep:
                return tf.concat([outputs[-1], G], axis=-1)
            else:
                return G


class MVMNet(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.factor_size = self.flags.factor_size

    def build(self, input_shape):
        self.num_features = input_shape[1]
        self.bias = self.add_variable("padding_bias",
                                      (self.num_features, self.factor_size))

    def call(self, inputs, **kwargs):
        """
        shape: Batch x Features x factor_size
        """
        embs = tf.stack(inputs, 1)

        all_order = tf.add(embs, self.bias)
        mvm = all_order[:, 0, :]  # B x 1 x factor_size
        for i in range(1, self.num_features):
            mvm = tf.multiply(mvm, all_order[:, i, :])
        mvm = tf.reshape(mvm, shape=[-1, self.factor_size])
        return mvm


class DeepMultiplyNet(tf.keras.layers.Layer):
    def __init__(self, hiddens, activation, sparse):
        super().__init__()
        self.hiddens = hiddens
        self.activation = activation
        self.sparse = sparse

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        with tf.name_scope('deep_multiply'):
            inputs = tf.concat(inputs, -1)
            for i, hs in enumerate(self.hiddens):
                with tf.name_scope('deep'):
                    deep1 = DeepBlock(hidden=hs,
                                      activation=self.activation,
                                      prefix='deep_multiply1_{}'.format(i),
                                      sparse=self.sparse)(inputs)
                    deep2 = DeepBlock(hidden=hs,
                                      activation=self.activation,
                                      prefix='deep_multiply2_{}'.format(i),
                                      sparse=self.sparse)(inputs)
                    inputs = deep1 + deep2 + tf.multiply(deep1, deep2)
            return inputs
