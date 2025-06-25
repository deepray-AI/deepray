# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""Keras-based attention layer."""

from __future__ import absolute_import
from __future__ import division

# from __future__ import google_type_annotations
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
import tf_keras as keras

from deepray.layers import dense_einsum
from deepray.layers import masked_softmax


class Attention(keras.layers.Layer):
  """Attention layer.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-width vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  Attributes:
    num_heads: Number of attention heads.
    head_size: Size of each attention head.
    dropout: Dropout probability.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(
    self,
    num_heads,
    head_size,
    dropout_rate=0.0,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs,
  ):
    super(Attention, self).__init__(**kwargs)
    self._num_heads = num_heads
    self._head_size = head_size
    self._dropout_rate = dropout_rate
    self._kernel_initializer = keras.initializers.get(kernel_initializer)
    self._bias_initializer = keras.initializers.get(bias_initializer)
    self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = keras.constraints.get(kernel_constraint)
    self._bias_constraint = keras.constraints.get(bias_constraint)

    self._query_dense = dense_einsum.DenseEinsum(
      output_shape=(self._num_heads, self._head_size),
      kernel_initializer=self._kernel_initializer,
      bias_initializer=self._bias_initializer,
      kernel_regularizer=self._kernel_regularizer,
      bias_regularizer=self._bias_regularizer,
      activity_regularizer=self._activity_regularizer,
      kernel_constraint=self._kernel_constraint,
      bias_constraint=self._bias_constraint,
      name="query",
    )

    self._key_dense = dense_einsum.DenseEinsum(
      output_shape=(self._num_heads, self._head_size),
      kernel_initializer=self._kernel_initializer,
      bias_initializer=self._bias_initializer,
      kernel_regularizer=self._kernel_regularizer,
      bias_regularizer=self._bias_regularizer,
      activity_regularizer=self._activity_regularizer,
      kernel_constraint=self._kernel_constraint,
      bias_constraint=self._bias_constraint,
      name="key",
    )

    self._value_dense = dense_einsum.DenseEinsum(
      output_shape=(self._num_heads, self._head_size),
      kernel_initializer=self._kernel_initializer,
      bias_initializer=self._bias_initializer,
      kernel_regularizer=self._kernel_regularizer,
      bias_regularizer=self._bias_regularizer,
      activity_regularizer=self._activity_regularizer,
      kernel_constraint=self._kernel_constraint,
      bias_constraint=self._bias_constraint,
      name="value",
    )

    self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])

    self._dropout = keras.layers.Dropout(rate=self._dropout_rate)

  def get_config(self):
    config = {
      "num_heads": self._num_heads,
      "head_size": self._head_size,
      "dropout_rate": self._dropout_rate,
      "kernel_initializer": keras.initializers.serialize(self._kernel_initializer),
      "bias_initializer": keras.initializers.serialize(self._bias_initializer),
      "kernel_regularizer": keras.regularizers.serialize(self._kernel_regularizer),
      "bias_regularizer": keras.regularizers.serialize(self._bias_regularizer),
      "activity_regularizer": keras.regularizers.serialize(self._activity_regularizer),
      "kernel_constraint": keras.constraints.serialize(self._kernel_constraint),
      "bias_constraint": keras.constraints.serialize(self._bias_constraint),
    }
    base_config = super(Attention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    from_tensor = inputs[0]
    to_tensor = inputs[1]
    attention_mask = inputs[2] if len(inputs) == 3 else None

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self._key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self._value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)

    # Instead of using the einsum equation, we expand it into the below
    # equivalent equations.
    # `query_tensor` = [B, N, F, H]
    query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
    # `key_tensor` = [B, N, T, H]
    key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)

    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    return tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)


class CachedAttention(Attention):
  """Attention layer with cache used for auto-agressive decoding.

  Attributes:
    num_heads: Number of attention heads.
    head_size: Size of each attention head.
    **kwargs: Other keyword arguments inherit from `Attention` class.
  """

  def __init__(self, num_heads, head_size, **kwargs):
    super(CachedAttention, self).__init__(num_heads, head_size, **kwargs)

  def _update_cache(self, key_tensor, value_tensor, cache, decode_loop_step):
    """Updates cache states and gets full-length key/value tensors."""
    # Combines cached keys and values with new keys and values.
    if decode_loop_step is not None:
      # TPU special case.
      key_seq_dim = cache["key"].shape.as_list()[1]
      indices = tf.reshape(tf.one_hot(decode_loop_step, key_seq_dim, dtype=key_tensor.dtype), [1, key_seq_dim, 1, 1])
      key_tensor = cache["key"] + key_tensor * indices
      value_seq_dim = cache["value"].shape.as_list()[1]
      indices = tf.reshape(
        tf.one_hot(decode_loop_step, value_seq_dim, dtype=value_tensor.dtype), [1, value_seq_dim, 1, 1]
      )
      value_tensor = cache["value"] + value_tensor * indices
    else:
      key_tensor = tf.concat([tf.cast(cache["key"], key_tensor.dtype), key_tensor], axis=1)
      value_tensor = tf.concat([tf.cast(cache["value"], value_tensor.dtype), value_tensor], axis=1)

    # Update cache
    cache["key"] = key_tensor
    cache["value"] = value_tensor

    return key_tensor, value_tensor

  def call(self, inputs, decode_loop_step=None):
    from_tensor = inputs[0]
    to_tensor = inputs[1]
    attention_mask = inputs[2] if len(inputs) >= 3 else None
    cache = inputs[3] if len(inputs) >= 4 else None
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self._key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self._value_dense(to_tensor)

    if cache:
      key_tensor, value_tensor = self._update_cache(key_tensor, value_tensor, cache, decode_loop_step)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    return tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor), cache


class WindowAttention(keras.layers.Layer):
  """
  ## Window based multi-head self-attention

  Usually Transformers perform global self-attention, where the relationships between
  a token and all other tokens are computed. The global computation leads to quadratic
  complexity with respect to the number of tokens. Here, as the [original paper](https://arxiv.org/abs/2103.14030)
  suggests, we compute self-attention within local windows, in a non-overlapping manner.
  Global self-attention leads to quadratic computational complexity in the number of patches,
  whereas window-based self-attention leads to linear complexity and is easily scalable.
  """

  def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.window_size = window_size
    self.num_heads = num_heads
    self.scale = (dim // num_heads) ** -0.5
    self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias)
    self.dropout = keras.layers.Dropout(dropout_rate)
    self.proj = keras.layers.Dense(dim)

  def build(self, input_shape):
    num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
    self.relative_position_bias_table = self.add_weight(
      shape=(num_window_elements, self.num_heads),
      initializer=tf.initializers.Zeros(),
      trainable=True,
    )
    coords_h = np.arange(self.window_size[0])
    coords_w = np.arange(self.window_size[1])
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack(coords_matrix)
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += self.window_size[0] - 1
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)

    self.relative_position_index = tf.Variable(
      initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
    )

  def call(self, x, mask=None):
    _, size, channels = x.shape
    head_dim = channels // self.num_heads
    x_qkv = self.qkv(x)
    x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
    x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
    q = q * self.scale
    k = tf.transpose(k, perm=(0, 1, 3, 2))
    attn = q @ k

    num_window_elements = self.window_size[0] * self.window_size[1]
    relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
    relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
    relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
      nW = mask.get_shape()[0]
      mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
      attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size)) + mask_float
      attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
      attn = keras.activations.softmax(attn, axis=-1)
    else:
      attn = keras.activations.softmax(attn, axis=-1)
    attn = self.dropout(attn)

    x_qkv = attn @ v
    x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
    x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
    x_qkv = self.proj(x_qkv)
    x_qkv = self.dropout(x_qkv)
    return x_qkv
