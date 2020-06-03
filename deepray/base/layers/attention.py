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
import numpy as np
import tensorflow as tf

from deepray.base.layers.core import DeepNet, Linear


class Attention(tf.keras.layers.Layer):
    """Attention layer.

    Parameters
    ----------
    input_size : int
        Size of input.

    hidden_layers : iterable
        Hidden layer sizes.

    dropout : float
        Dropout rate.

    activation : str
        Name of activation function. relu, prelu and sigmoid are supported.

    return_scores : bool
        Return attention scores instead of weighted sum pooling result.
    """

    def __init__(
            self,
            hidden_layers,
            dropout=0.0,
            batchnorm=True,
            activation='prelu',
            return_scores=False):
        super(Attention, self).__init__()
        self.return_scores = return_scores
        self.mlp = DeepNet(hidden=hidden_layers, droprate=dropout, activation=activation, batchnorm=batchnorm)
        self.fc = Linear(1)

    def call(self, query, keys, keys_length):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        kerys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, if return_scores=False [B, H], otherwise [B, T]
        """
        batch_size, max_length, dim = keys.size()

        query = query.unsqueeze(1).expand(-1, max_length, -1)

        din_all = tf.concat(
            [query, keys, query - keys, query * keys], dim=-1)

        din_all = din_all.view(batch_size * max_length, -1)

        outputs = self.mlp(din_all)

        outputs = self.fc(outputs).view(batch_size, max_length)  # [B, T]

        # Scale
        outputs = outputs / (dim ** 0.5)

        # Mask
        mask = (torch.arange(max_length, device=keys_length.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1))
        outputs[~mask] = -np.inf

        # Activation
        outputs = tf.nn.softmax(outputs, dim=1)  # [B, T]

        if not self.return_scores:
            # Weighted sum
            outputs = tf.matmul(
                outputs.unsqueeze(1), keys).squeeze()  # [B, H]

        return outputs


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class LocalActivationUnit(tf.keras.layers.Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), activation='sigmoid',
                 l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=tf.keras.initializers.glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="bias")
        self.dnn = DeepNet(hidden=self.hidden_units,
                           activation=self.activation,
                           droprate=self.dropout_rate,
                           batchnorm=self.use_bn)

        self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
            x[0], x[1], axes=(-1, 0)), x[2]))

    def call(self, query, user_behavior, is_training=None, **kwargs):
        keys_len = user_behavior.get_shape()[0]
        queries = tf.keras.backend.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, user_behavior, queries - user_behavior, queries * user_behavior], axis=-1)

        att_out = self.dnn(att_input, training=is_training)

        attention_score = self.dense([att_out, self.kernel, self.bias])

        return attention_score


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wide_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def point_wide_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
