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
"""Implements the losses for TF-Ranking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
from typing import Callable, Dict, Tuple
import tensorflow as tf

_PADDING_LABEL = -1.0
_PADDING_PREDICTION = -1e6
_PADDING_WEIGHT = 0.0

TensorLike = tf.types.experimental.TensorLike
TransformationFunction = Callable[[TensorLike], tf.Tensor]
LossFunction = Callable[[TensorLike, TensorLike, Dict[str, TensorLike]], tf.Tensor]
MetricFunction = Callable[[TensorLike, TensorLike, Dict[str, TensorLike]], tf.Tensor]


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


class _RankingLoss(object, metaclass=abc.ABCMeta):
  """Interface for ranking loss."""

  def __init__(self, name, lambda_weight=None, temperature=1.0, ragged=False):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    self._name = name
    self._lambda_weight = lambda_weight
    self._temperature = temperature
    self._ragged = ragged

  @property
  def name(self):
    """The loss name."""
    return self._name

  def _prepare_and_validate_params(self, labels, logits, weights, mask):
    """Prepares and validate input parameters.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      A tuple (labels, logits, weights, mask) of `tf.Tensor` objects that are
      ready to be used in the loss.
    """
    if self._ragged:
      labels, logits, weights, mask = ragged_to_dense(labels, logits, weights)

    if mask is None:
      mask = is_label_valid(labels)

    if weights is None:
      weights = 1.0

    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    weights = tf.convert_to_tensor(weights)
    mask = tf.convert_to_tensor(mask)

    return labels, logits, weights, mask

  def compute_unreduced_loss(self, labels, logits, mask=None):
    """Computes the unreduced loss.

    Args:
      labels: A `Tensor` or `RaggedTensor` of the same shape as `logits`
        representing graded relevance.
      logits: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size].
        Each value is the ranking score of the corresponding item.
      mask: An optional `Tensor` of the same shape as logits indicating which
        entries are valid for computing the loss. Will be ignored if the loss
        was constructed with ragged=True.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    labels, logits, _, mask = self._prepare_and_validate_params(labels, logits, None, mask)
    return self._compute_unreduced_loss_impl(labels, logits, mask)

  @abc.abstractmethod
  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """Implementation for the unreduced loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      mask: An optional `Tensor` of the same shape as logits indicating which
        entries are valid for computing the loss.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    raise NotImplementedError("Calling an abstract method.")

  def normalize_weights(self, labels, weights):
    """Normalizes weights.

    This is needed for `tf.estimator` given that the reduction may be
    `SUM_OVER_NONZERO_WEIGHTS`.

    This method is also needed to compute normalized weights when calling
    `compute_unreduced_loss`, which is done in the tf.keras losses.

    Args:
      labels: A `Tensor` of shape [batch_size, list_size] representing graded
        relevance.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      The normalized weights.
    """
    if self._ragged:
      labels, _, weights, _ = utils.ragged_to_dense(labels, None, weights)
    return self._normalize_weights_impl(labels, weights)

  def _normalize_weights_impl(self, labels, weights):
    """See `normalize_weights`."""
    del labels
    return 1.0 if weights is None else weights

  def get_logits(self, logits):
    """Computes logits rescaled by temperature.

    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.

    Returns:
      Tensor of rescaled logits.
    """
    if not tf.is_tensor(logits):
      logits = tf.convert_to_tensor(value=logits)
    return logits / self._temperature

  def compute(self, labels, logits, weights, reduction, mask=None):
    """Computes the reduced loss for tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      Reduced loss for training and eval.
    """
    logits = self.get_logits(logits)
    losses, loss_weights = self._compute_unreduced_loss_impl(labels, logits, mask)
    weights = tf.multiply(self._normalize_weights_impl(labels, weights), loss_weights)
    return tf.compat.v1.losses.compute_weighted_loss(losses, weights, reduction=reduction)

  @abc.abstractmethod
  def compute_per_list(self, labels, logits, weights, mask=None):
    """Computes the per-list loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      A pair of `Tensor` objects of shape [batch_size] containing per-list
      losses and weights.
    """
    raise NotImplementedError("Calling an abstract method.")

  def eval_metric(self, labels, logits, weights, mask=None):
    """Computes the eval metric for the loss in tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the metric.

    Returns:
      A metric op.
    """
    losses, loss_weights = self._compute_unreduced_loss_impl(labels, logits, mask)
    weights = tf.multiply(self._normalize_weights_impl(labels, weights), loss_weights)
    return tf.compat.v1.metrics.mean(losses, weights)


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


def is_label_valid(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = tf.convert_to_tensor(value=labels)
  return tf.greater_equal(labels, 0.0)
