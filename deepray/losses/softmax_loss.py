from typing import Any, Dict, Optional

import tensorflow as tf

from deepray.losses import losses_impl
from deepray.losses import utils

# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


class _RankingLoss(tf.keras.losses.Loss):
  """Base class for all ranking losses.

  Please see tf.keras.losses.Loss for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(
    self, reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO, name: Optional[str] = None, ragged: bool = False
  ):
    super().__init__(reduction, name)
    # An instance of loss in `losses_impl`. Overwrite this in subclasses.
    self._loss = None
    self._ragged = ragged

  def __call__(
    self, y_true: utils.TensorLike, y_pred: utils.TensorLike, sample_weight: Optional[utils.TensorLike] = None
  ) -> tf.Tensor:
    """See tf.keras.losses.Loss."""
    if self._loss is None:
      raise ValueError("self._loss is not defined. Please use a subclass.")
    sample_weight = self._loss.normalize_weights(y_true, sample_weight)
    return super().__call__(y_true, y_pred, sample_weight)

  def call(self, y_true: utils.TensorLike, y_pred: utils.TensorLike) -> tf.Tensor:
    """See tf.keras.losses.Loss."""
    y_pred = self._loss.get_logits(y_pred)
    losses, weights = self._loss.compute_unreduced_loss(labels=y_true, logits=y_pred)
    return tf.multiply(losses, weights)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({"ragged": self._ragged})
    return config


class _ListwiseLoss(_RankingLoss):
  """Base class for listwise ranking losses."""

  def __init__(
    self,
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    lambda_weight: Optional[losses_impl._LambdaWeight] = None,
    temperature: float = 1.0,
    ragged: bool = False,
    **kwargs,
  ):
    super().__init__(reduction, name, ragged)
    self._lambda_weight = lambda_weight
    self._temperature = temperature

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
      "lambda_weight": utils.serialize_keras_object(self._lambda_weight),
      "temperature": self._temperature,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    config.update({
      "lambda_weight": utils.deserialize_keras_object(config["lambda_weight"]),
    })
    return cls(**config)


class SoftmaxLoss(_ListwiseLoss):
  r"""Computes Softmax cross-entropy loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = - sum_i y_i * log(softmax(s_i))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = dp.losses.SoftmaxLoss()
  >>> loss(y_true, y_pred).numpy()
  0.7981389

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = dp.losses.SoftmaxLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.83911896

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.SoftmaxLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \sum_i y_i
  \log\left(\frac{\exp(s_i)}{\sum_j \exp(s_j)}\right)
  $$
  """

  def __init__(
    self,
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    lambda_weight: Optional[losses_impl._LambdaWeight] = None,
    temperature: float = 1.0,
    ragged: bool = False,
  ):
    """Softmax cross-entropy loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.SoftmaxLoss(
      name="{}_impl".format(name) if name else None, lambda_weight=lambda_weight, temperature=temperature, ragged=ragged
    )

  def __call__(
    self, y_true: utils.TensorLike, y_pred: utils.TensorLike, sample_weight: Optional[utils.TensorLike] = None
  ) -> tf.Tensor:
    """See _RankingLoss."""
    losses, sample_weight = self._loss.compute_per_list(y_true, y_pred, sample_weight)
    return tf.keras.__internal__.losses.compute_weighted_loss(losses, sample_weight, reduction=self._get_reduction())
