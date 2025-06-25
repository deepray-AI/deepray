"""NCF model input pipeline."""

import os
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline
from deepray.datasets.movielens import constants as rconst


class Produce(DataPipeline):
  def __init__(self, params, producer):
    self._producer = producer
    self._params = params

  def preprocess_train_input(self, features, labels):
    """Pre-process the training data.

    This is needed because
    - The label needs to be extended to be used in the loss fn
    - We need the same inputs for training and eval so adding fake inputs
      for DUPLICATE_MASK in training data.

    Args:
      features: Dictionary of features for training.
      labels: Training labels.

    Returns:
      Processed training features.
    """
    fake_dup_mask = tf.zeros_like(features[rconst.USER_COLUMN])
    features[rconst.DUPLICATE_MASK] = fake_dup_mask
    features[rconst.TRAIN_LABEL_KEY] = labels
    return features, labels

  def preprocess_eval_input(self, features):
    """Pre-process the eval data.

    This is needed because:
    - The label needs to be extended to be used in the loss fn
    - We need the same inputs for training and eval so adding fake inputs
      for VALID_PT_MASK in eval data.

    Args:
      features: Dictionary of features for evaluation.

    Returns:
      Processed evaluation features.
    """
    labels = tf.cast(tf.zeros_like(features[rconst.USER_COLUMN]), tf.bool)
    fake_valid_pt_mask = tf.cast(tf.zeros_like(features[rconst.USER_COLUMN]), tf.bool)
    features[rconst.VALID_POINT_MASK] = fake_valid_pt_mask
    features[rconst.TRAIN_LABEL_KEY] = labels
    return features, labels

  def build_dataset(
    self, input_file_pattern, batch_size, is_training=True, context=None, use_horovod=False, *args, **kwargs
  ):
    input_fn = self._producer.make_input_fn(is_training=is_training)
    if is_training:
      input_dataset = input_fn(self._params).map(self.preprocess_train_input)
    else:
      input_dataset = input_fn(self._params).map(self.preprocess_eval_input)
    return input_dataset
