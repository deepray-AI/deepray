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
from tensorflow.keras.layers import Dense, Activation


class CIN(tf.keras.layers.Layer):
    def __init__(self, hidden, activation, use_bias, use_res, use_direct, use_reduce_D):
        super(CIN, self).__init__()
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_res
        self.direct = use_direct
        self.reduce_D = use_reduce_D
        self.hidden = hidden
        if len(self.hidden) == 0:
            raise ValueError(
                "cin_layer_size must be a list(tuple) of length greater than 1")

    def build(self, input_shape):
        self.field_nums = [len(input_shape)]
        embed_dim = input_shape[0][1]
        self.f_ = []
        self.filters0 = []
        self.f__ = []
        self.bias = []

        for i, layer_size in enumerate(self.hidden):
            if self.reduce_D:
                self.filters0.append(
                    self.add_weight(name=f'f0_{i}', shape=[1, layer_size, self.field_nums[0], embed_dim],
                                    dtype=tf.float32, initializer='he_uniform'))
                self.f__.append(
                    self.add_weight(name=f'f__{i}', shape=[1, layer_size, embed_dim, self.field_nums[-1]],
                                    dtype=tf.float32, initializer='he_uniform'))
            else:
                self.f_.append(
                    self.add_weight(name=f'f_{i}', shape=[1, self.field_nums[-1] * self.field_nums[0], layer_size],
                                    dtype=tf.float32, initializer='he_uniform'))
            if self.use_bias:
                self.bias.append(self.add_weight(name=f'bias{i}', shape=[layer_size], dtype=tf.float32,
                                                 initializer='zeros'))

            if self.direct:
                self.field_nums.append(layer_size)
            else:
                if i != len(self.hidden) - 1 and layer_size % 2 > 0:
                    raise ValueError(
                        "cin_layer_size must be even number except for the last layer when direct=True")
                self.field_nums.append(layer_size // 2)
        self.activation_layers = [Activation(self.activation) for _ in self.hidden]
        if self.use_residual:
            self.exFM_out0 = Dense(self.hidden[-1], activation=self.activation,
                                   kernel_initializer='he_uniform')
        super(CIN, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Reshape to Batch_size x num_fields x embdding_size
        """
        x = tf.concat([tf.expand_dims(key, axis=1) for key in x], 1)

        dim = int(x.get_shape()[-1])
        hidden_nn_layers = [x]
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.hidden):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            """
            dot_result_m shape :  (Dim, Batch, FieldNum, HiddenNum), a.k.a (D,B,F,H)
            """
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m,
                shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])  # shape: (D,B,FH)
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # (B,D,FH)

            if self.reduce_D:
                f0_ = self.filters0[idx]
                f__ = self.f__[idx]
                f_m = tf.matmul(f0_, f__)
                f_o = tf.reshape(f_m, shape=[1, layer_size, self.field_nums[0] * self.field_nums[idx]])
                filters = tf.transpose(f_o, perm=[0, 2, 1])
            else:
                filters = self.f_[idx]
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
            if self.use_bias:
                curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.direct:
                direct_connect = curr_out
                next_hidden = curr_out
            else:
                if idx != len(self.hidden) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)

        if self.use_residual:
            exFM_out0 = self.exFM_out0(result)
            result = tf.concat([exFM_out0, result], axis=1)
        return result

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(CIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
