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
import tensorflow as tf
from tensorflow.keras.initializers import (glorot_uniform)
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2


class CompressedInteractionNetwork(tf.keras.layers.Layer):
    def __init__(self, layer_size, activation, split_half, use_res):
        super(CompressedInteractionNetwork, self).__init__()
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = 1e-5
        self.seed = 1024
        self.use_residual = use_res
        if len(self.layer_size) == 0:
            raise ValueError(
                "cin_layer_size must be a list(tuple) of length greater than 1")

    def build(self, input_shape):
        self.field_nums = [input_shape[1]]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):
            self.filters.append(
                self.add_weight(name='filter_{}'.format(i), shape=[1, self.field_nums[-1] * self.field_nums[0], size],
                                dtype=tf.float32,
                                initializer=glorot_uniform(seed=self.seed),
                                regularizer=l2(self.l2_reg)))
            self.bias.append(self.add_weight(name='bias_{}'.format(i), shape=[size], dtype=tf.float32,
                                             initializer='zeros'))
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError("cin_layer_size must be even number except for the last layer when split_half=True")
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        self.activation_layers = [Activation(self.activation) for _ in self.layer_size]

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m,
                shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(dot_result, filters=self.filters[idx], stride=1, padding='VALID')
            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        logit = tf.concat(final_result, axis=1)
        logit = tf.reduce_sum(logit, -1, keepdims=False)
        return logit

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], featuremap_num])
