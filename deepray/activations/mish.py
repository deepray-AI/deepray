# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

from deepray.utils.types import TensorLike


@tf.keras.utils.register_keras_serializable(package="Deepray")
def mish(x: TensorLike) -> tf.Tensor:
  r"""Mish: A Self Regularized Non-Monotonic Neural Activation Function.

  Computes mish activation:

  $$
  \mathrm{mish}(x) = x \cdot \tanh(\mathrm{softplus}(x)).
  $$

  See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).

  Usage:

  >>> x = tf.constant([1.0, 0.0, 1.0])
  >>> dp.activations.mish(x)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.865098..., 0.       , 0.865098...], dtype=float32)>

  Args:
      x: A `Tensor`. Must be one of the following types:
          `bfloat16`, `float16`, `float32`, `float64`.
  Returns:
      A `Tensor`. Has the same type as `x`.
  """
  x = tf.convert_to_tensor(x)
  return x * tf.math.tanh(tf.math.softplus(x))
