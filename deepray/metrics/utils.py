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
"""Utilities for metrics."""

from typing import Callable
from typing import Optional

import numpy as np
import tensorflow as tf
from typeguard import typechecked

from deepray.utils.types import AcceptableDTypes

_PADDING_LABEL = -1.0
_PADDING_PREDICTION = -1e6
_PADDING_WEIGHT = 0.0

TensorLike = tf.types.experimental.TensorLike
GainFunction = Callable[[TensorLike], tf.Tensor]
RankDiscountFunction = Callable[[TensorLike], tf.Tensor]
PositiveFunction = Callable[[TensorLike], tf.Tensor]


class MeanMetricWrapper(tf.keras.metrics.Mean):
  """Wraps a stateless metric function with the Mean metric."""

  @typechecked
  def __init__(
    self,
    fn: Callable,
    name: Optional[str] = None,
    dtype: AcceptableDTypes = None,
    **kwargs,
  ):
    """Creates a `MeanMetricWrapper` instance.
    Args:
      fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super().__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.
    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1.
        Can be a `Tensor` whose rank is either 0, or the same rank as
        `y_true`, and must be broadcastable to `y_true`.
    Returns:
      Update op.
    """
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)
    # TODO: Add checks for ragged tensors and dimensions:
    #   `ragged_assert_compatible_and_get_flat_values`
    #   and `squeeze_or_expand_dimensions`
    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super().update_state(matches, sample_weight=sample_weight)

  def get_config(self):
    config = {k: v for k, v in self._fn_kwargs.items()}
    base_config = super().get_config()
    return {**base_config, **config}


def _get_model(metric, num_output):
  # Test API comptibility with tf.keras Model
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(64, activation="relu"))
  model.add(tf.keras.layers.Dense(num_output, activation="softmax"))
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc", metric])

  data = np.random.random((10, 3))
  labels = np.random.random((10, num_output))
  model.fit(data, labels, epochs=1, batch_size=5, verbose=0)


def sample_weight_shape_match(v, sample_weight):
  if sample_weight is None:
    return tf.ones_like(v)
  if np.size(sample_weight) == 1:
    return tf.fill(v.shape, sample_weight)
  return tf.convert_to_tensor(sample_weight)


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


def is_label_valid(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = tf.convert_to_tensor(value=labels)
  return tf.greater_equal(labels, 0.0)


def _get_shuffle_indices(shape, mask=None, shuffle_ties=True, seed=None):
  """Gets indices which would shuffle a tensor.

  Args:
    shape: The shape of the indices to generate.
    mask: An optional mask that indicates which entries to place first. Its
      shape should be equal to given shape.
    shuffle_ties: Whether to randomly shuffle ties.
    seed: The ops-level random seed.

  Returns:
    An int32 `Tensor` with given `shape`. Its entries are indices that would
    (randomly) shuffle the values of a `Tensor` of given `shape` along the last
    axis while placing masked items first.
  """
  # Generate random values when shuffling ties or all zeros when not.
  if shuffle_ties:
    shuffle_values = tf.random.uniform(shape, seed=seed)
  else:
    shuffle_values = tf.zeros(shape, dtype=tf.float32)

  # Since shuffle_values is always in [0, 1), we can safely increase entries
  # where mask=False with 2.0 to make sure those are placed last during the
  # argsort op.
  if mask is not None:
    shuffle_values = tf.where(mask, shuffle_values, shuffle_values + 2.0)

  # Generate indices by sorting the shuffle values.
  return tf.argsort(shuffle_values, stable=True)


def sort_by_scores(scores, features_list, topn=None, shuffle_ties=True, seed=None, mask=None):
  """Sorts list of features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.
    mask: An optional `Tensor` of shape [batch_size, list_size] representing
      which entries are valid for sorting. Invalid entries will be pushed to the
      end.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  with tf.compat.v1.name_scope(name="sort_by_scores"):
    scores = tf.cast(scores, tf.float32)
    scores.get_shape().assert_has_rank(2)
    list_size = tf.shape(input=scores)[1]
    if topn is None:
      topn = list_size
    topn = tf.minimum(topn, list_size)

    # Set invalid entries (those whose mask value is False) to the minimal value
    # of scores so they will be placed last during sort ops.
    if mask is not None:
      scores = tf.where(mask, scores, tf.reduce_min(scores))

    # Shuffle scores to break ties and/or push invalid entries (according to
    # mask) to the end.
    shuffle_ind = None
    if shuffle_ties or mask is not None:
      shuffle_ind = _get_shuffle_indices(tf.shape(input=scores), mask, shuffle_ties=shuffle_ties, seed=seed)
      scores = tf.gather(scores, shuffle_ind, batch_dims=1, axis=1)

    # Perform sort and return sorted feature_list entries.
    _, indices = tf.math.top_k(scores, topn, sorted=True)
    if shuffle_ind is not None:
      indices = tf.gather(shuffle_ind, indices, batch_dims=1, axis=1)
    return [tf.gather(f, indices, batch_dims=1, axis=1) for f in features_list]


def ragged_to_dense(labels, predictions, weights):
  """Converts given inputs from ragged tensors to dense tensors.

  Args:
    labels: A `tf.RaggedTensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
      value is the ranking score of the corresponding example.
    weights: An optional `tf.RaggedTensor` of the same shape of predictions or a
      `tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
      the latter case is per-list.

  Returns:
    A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
  """
  # TODO: Add checks to validate (ragged) shapes of input tensors.
  mask = tf.cast(tf.ones_like(labels).to_tensor(0.0), dtype=tf.bool)
  labels = labels.to_tensor(_PADDING_LABEL)
  if predictions is not None:
    predictions = predictions.to_tensor(_PADDING_PREDICTION)
  if isinstance(weights, tf.RaggedTensor):
    weights = weights.to_tensor(_PADDING_WEIGHT)
  return labels, predictions, weights, mask
