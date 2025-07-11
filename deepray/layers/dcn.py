# Copyright 2023 The TensorFlow Recommenders Authors.
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
"""Implements `Cross` Layer, the cross layer in Deep & Cross Network (DCN)."""

from typing import Union, Text, Optional

import tensorflow as tf
import tf_keras as keras


class Cross(keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions.

  A layer that creates explicit and bounded-degree feature interactions
  efficiently. The `call` method accepts `inputs` as a tuple of size 2
  tensors. The first input `x0` is the base layer that contains the original
  features (usually the embedding layer); the second input `xi` is the output
  of the previous `Cross` layer in the stack, i.e., the i-th `Cross`
  layer. For the first `Cross` layer in the stack, x0 = xi.

  The output is x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
  where .* designates elementwise multiplication, W could be a full-rank
  matrix, or a low-rank matrix U*V to reduce the computational cost, and
  diag_scale increases the diagonal of W to improve training stability (
  especially for the low-rank case).

  References:
      1. [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf)
        See Eq. (1) for full-rank and Eq. (2) for low-rank version.
      2. [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)

  Example:

      ```python
      # after embedding layer in a functional model:
      input = keras.Input(shape=(None,), name='index', dtype=tf.int64)
      x0 = dp.layers.Embedding(vocabulary_size=32, embedding_dim=6)
      x1 = Cross()(x0, x0)
      x2 = Cross()(x0, x1)
      logits = keras.layers.Dense(units=10)(x2)
      model = keras.Model(input, logits)
      ```

  Args:
      projection_dim: project dimension to reduce the computational cost.
        Default is `None` such that a full (`input_dim` by `input_dim`) matrix
        W is used. If enabled, a low-rank matrix W = U*V will be used, where U
        is of size `input_dim` by `projection_dim` and V is of size
        `projection_dim` by `input_dim`. `projection_dim` need to be smaller
        than `input_dim`/2 to improve the model efficiency. In practice, we've
        observed that `projection_dim` = d/4 consistently preserved the
        accuracy of a full-rank version.
      diag_scale: a non-negative float used to increase the diagonal of the
        kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an
        identity matrix.
      use_bias: whether to add a bias term for this layer. If set to False,
        no bias term will be used.
      preactivation: Activation applied to output matrix of the layer, before
        multiplication with the input. Can be used to control the scale of the
        layer's outputs and improve stability.
      kernel_initializer: Initializer to use on the kernel matrix.
      bias_initializer: Initializer to use on the bias vector.
      kernel_regularizer: Regularizer to use on the kernel matrix.
      bias_regularizer: Regularizer to use on bias vector.

  Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
  Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

  def __init__(
    self,
    projection_dim: Optional[int] = None,
    diag_scale: Optional[float] = 0.0,
    use_bias: bool = True,
    preactivation: Optional[Union[str, keras.layers.Activation]] = None,
    kernel_initializer: Union[Text, keras.initializers.Initializer] = "truncated_normal",
    bias_initializer: Union[Text, keras.initializers.Initializer] = "zeros",
    kernel_regularizer: Union[Text, None, keras.regularizers.Regularizer] = None,
    bias_regularizer: Union[Text, None, keras.regularizers.Regularizer] = None,
    **kwargs,
  ):
    super(Cross, self).__init__(**kwargs)

    self._projection_dim = projection_dim
    self._diag_scale = diag_scale
    self._use_bias = use_bias
    self._preactivation = keras.activations.get(preactivation)
    self._kernel_initializer = keras.initializers.get(kernel_initializer)
    self._bias_initializer = keras.initializers.get(bias_initializer)
    self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = keras.regularizers.get(bias_regularizer)
    self._input_dim = None

    self._supports_masking = True

    if self._diag_scale < 0:  # pytype: disable=unsupported-operands
      raise ValueError("`diag_scale` should be non-negative. Got `diag_scale` = {}".format(self._diag_scale))

  def build(self, input_shape):
    last_dim = input_shape[-1]

    if self._projection_dim is None:
      self._dense = keras.layers.Dense(
        last_dim,
        kernel_initializer=_clone_initializer(self._kernel_initializer),
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias,
        dtype=self.dtype,
        activation=self._preactivation,
      )
    else:
      self._dense_u = keras.layers.Dense(
        self._projection_dim,
        kernel_initializer=_clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        use_bias=False,
        dtype=self.dtype,
      )
      self._dense_v = keras.layers.Dense(
        last_dim,
        kernel_initializer=_clone_initializer(self._kernel_initializer),
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias,
        dtype=self.dtype,
        activation=self._preactivation,
      )
    self.built = True

  def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Computes the feature cross.

    Args:
      x0: The input tensor
      x: Optional second input tensor. If provided, the layer will compute
        crosses between x0 and x; if not provided, the layer will compute
        crosses between x0 and itself.

    Returns:
     Tensor of crosses.
    """

    if not self.built:
      self.build(x0.shape)

    if x is None:
      x = x0

    if x0.shape[-1] != x.shape[-1]:
      raise ValueError(
        "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
        "dimension {}. This case is not supported yet.".format(x0.shape[-1], x.shape[-1])
      )

    if self._projection_dim is None:
      prod_output = self._dense(x)
    else:
      prod_output = self._dense_v(self._dense_u(x))

    prod_output = tf.cast(prod_output, self.compute_dtype)

    if self._diag_scale:
      prod_output = prod_output + self._diag_scale * x

    return x0 * prod_output + x

  def get_config(self):
    config = {
      "projection_dim": self._projection_dim,
      "diag_scale": self._diag_scale,
      "use_bias": self._use_bias,
      "preactivation": keras.activations.serialize(self._preactivation),
      "kernel_initializer": keras.initializers.serialize(self._kernel_initializer),
      "bias_initializer": keras.initializers.serialize(self._bias_initializer),
      "kernel_regularizer": keras.regularizers.serialize(self._kernel_regularizer),
      "bias_regularizer": keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _clone_initializer(initializer):
  return initializer.__class__.from_config(initializer.get_config())
