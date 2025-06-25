# Copyright 2024 The TensorFlow Ranking Authors.
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

import tensorflow as tf
from typing import Callable

TensorLike = tf.types.experimental.TensorLike
GainFunction = Callable[[TensorLike], tf.Tensor]
RankDiscountFunction = Callable[[TensorLike], tf.Tensor]
PositiveFunction = Callable[[TensorLike], tf.Tensor]


class _RankingMetric(tf.keras.metrics.Mean):
  """Implements base ranking metric class.

  Please see tf.keras.metrics.Mean for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(_RankingMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    # An instance of `metrics_impl._RankingMetric`.
    # Overwrite this in subclasses.
    self._metric = None
    self._ragged = ragged

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)

    # TODO: Add mask argument for metric.compute() call
    per_list_metric_val, per_list_metric_weights = self._metric.compute(y_true, y_pred, sample_weight)
    return super(_RankingMetric, self).update_state(per_list_metric_val, sample_weight=per_list_metric_weights)

  def get_config(self):
    config = super(_RankingMetric, self).get_config()
    config.update({
      "ragged": self._ragged,
    })
    return config


def serialize_keras_object(obj):
  if hasattr(tf.keras.utils, "legacy"):
    return tf.keras.utils.legacy.serialize_keras_object(obj)
  else:
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize_keras_object(config, module_objects=None, custom_objects=None, printable_module_name=None):
  if hasattr(tf.keras.utils, "legacy"):
    return tf.keras.utils.legacy.deserialize_keras_object(config, custom_objects, module_objects, printable_module_name)
  else:
    return tf.keras.utils.deserialize_keras_object(config, custom_objects, module_objects, printable_module_name)


# The following functions are used to transform labels and ranks for losses and
# metrics computation. User customized functions can be defined similarly by
# following the same annotations.
def identity(label: TensorLike) -> tf.Tensor:
  """Identity function that returns the input label.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    The input label.
  """
  return label


def inverse(rank: TensorLike) -> tf.Tensor:
  """Computes the inverse of input rank.

  Args:
    rank: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `1/x`.
  """
  return tf.math.divide_no_nan(1.0, rank)


def pow_minus_1(label: TensorLike) -> tf.Tensor:
  """Computes `2**x - 1` element-wise for each label.

  Can be used to define `gain_fn` for `tfr.keras.metrics.NDCGMetric`.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `2**x - 1`.
  """
  return tf.math.pow(2.0, label) - 1.0


def log2_inverse(rank: TensorLike) -> tf.Tensor:
  """Computes `1./log2(1+x)` element-wise for each label.

  Can be used to define `rank_discount_fn` for `tfr.keras.metrics.NDCGMetric`.

  Args:
    rank: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `1./log2(1+x)`.
  """
  return tf.math.divide_no_nan(tf.math.log(2.0), tf.math.log1p(rank))


def is_greater_equal_1(label: TensorLike) -> tf.Tensor:
  """Computes whether label is greater or equal to 1.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `I(x > 1)`.
  """
  return tf.greater_equal(label, 1.0)


def symmetric_log1p(t: TensorLike) -> tf.Tensor:
  """Computes `sign(x) * log(1 + sign(x))`.

  Args:
    t: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `I(x > 1)`.
  """
  return tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
