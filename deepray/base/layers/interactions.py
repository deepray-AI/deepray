#  Copyright © 2020-2020 Hailin Fu All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
"""
Base layers for learning feature interactions

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import tensorflow as tf

from deepray.base.layers.core import CustomDropout, DeepBlock, Linear


class FMNet(tf.keras.layers.Layer):
    def __init__(self, k=tf.constant(10), **kwargs):
        super().__init__()
        self.k = k

    def build(self, input_shape):
        n, p = input_shape
        self.linear_block = Linear()
        # interaction factors, randomly initialized
        self.kernel = self.add_weight(name='fm_kernel',
                                      shape=(self.k, p),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, inputs, **kwargs):
        # Calculate output with FM equation
        linear_terms = self.linear_block(inputs)
        pair_interactions = tf.multiply(tf.constant(0.5),
                                        tf.math.reduce_sum(
                                            tf.subtract(
                                                tf.pow(tf.matmul(inputs, tf.transpose(self.kernel)), 2),
                                                tf.matmul(tf.pow(inputs, 2), tf.transpose(tf.pow(self.kernel, 2)))),
                                            1, keepdims=True))
        logit = tf.add(linear_terms, pair_interactions)
        return logit


class CrossBlock(tf.keras.layers.Layer):
    def __init__(self, block_name, use_bias, sparse, flags):
        super(CrossBlock, self).__init__()
        self.use_bias = use_bias
        self.sparse = sparse
        self.block_name = block_name
        self.flags = flags

    def build(self, input_shape):
        self.kernel = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.flags.l1,
                                                           l2=self.flags.l2)
            if self.sparse else None,
            kernel_initializer=tf.zeros_initializer()
            if self.sparse else None,
            name='{}_weight'.format(self.block_name),
            use_bias=self.use_bias)

    def call(self, x0, x, is_training=None):
        outputs = self.kernel(x)
        if self.flags.summary_mode == 'all':
            for weight in tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                    scope=self.kernel.scope_name):
                tf.summary.histogram(
                    "normal/{}".format(weight.name), weight)
                nnz = tf.math.count_nonzero(weight)
                tf.summary.scalar('nnz/{}'.format(weight.name), nnz)
        return outputs * x0


class CrossNet(tf.keras.layers.Layer):
    def __init__(self, num_layers, use_bias=True, sparse=False, flags=None):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.sparse = sparse
        self.flags = flags

    def build(self, input_shape):
        self.kernel = [CrossBlock('{}'.format(i), self.use_bias, self.sparse, self.flags) for i in
                       range(self.num_layers)]

    def call(self, inputs, is_training=None):
        # inputs = tf.concat(raw_inputs, -1)
        t = inputs
        for i in range(self.num_layers):
            t += self.kernel[i](inputs, t, is_training=is_training)
        if self.flags.summary_mode == 'all':
            tf.summary.histogram(t.name, t)
        return t


class multihead_attention(tf.keras.layers.Layer):
    def __init__(self, num_units=None, num_heads=1, dropout_keep_prob=1, has_residual=True):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_keep_prob = dropout_keep_prob
        self.has_residual = has_residual
        super(multihead_attention, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, queries, keys, values, is_training=True, **kwargs):
        if self.num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.keras.layers.Dense(self.num_units, activation=tf.nn.relu)(queries)
        K = tf.keras.layers.Dense(self.num_units, activation=tf.nn.relu)(keys)
        V = tf.keras.layers.Dense(self.num_units, activation=tf.nn.relu)(values)
        if self.has_residual:
            V_res = tf.keras.layers.Dense(self.num_units, activation=tf.nn.relu)(values)
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        weights = tf.nn.softmax(weights)
        # Dropouts
        weights = CustomDropout(rate=1 - self.dropout_keep_prob)(weights, is_training)

        # Weighted sum
        outputs = tf.matmul(weights, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
        # Residual connection
        if self.has_residual:
            outputs += V_res
        outputs = tf.nn.relu(outputs)
        # Normalize
        outputs = self.normalize(outputs)
        return outputs

    def normalize(self, inputs, epsilon=1e-8):
        """
        Applies layer normalization
        Args:
            inputs: A tensor with 2 or more dimensions
            epsilon: A floating number to prevent Zero Division
        Returns:
            A tensor with the same shape and data dtype
        """
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

        return outputs


class SelfAttentionNet(tf.keras.layers.Layer):
    def __init__(self, hidden, concat_last_deep):
        super(SelfAttentionNet, self).__init__()
        self.hidden = hidden
        self.concat_last_deep = concat_last_deep

    def build(self, input_shape):
        self.kernel = [DeepBlock(hs,
                                 tf.nn.relu,
                                 'deep_{}'.format(i)) for i, hs in enumerate(self.hidden)]

    def call(self, raw_inputs, **kwargs):
        outputs = []
        with tf.name_scope('deep'):
            output = tf.concat(raw_inputs, -1)
            for i, hs in enumerate(self.hidden):
                output = CustomDropout(0.1)(output)
                output = self.kernel[i](output)
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
