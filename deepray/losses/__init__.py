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
"""Additional losses that conform to Keras API."""

import abc

from absl import flags
import tensorflow as tf
from packaging.version import parse

if parse(tf.__version__) < parse("2.11"):
  from keras.engine import compile_utils
elif parse(tf.__version__) > parse("2.16.0"):
  from tf_keras.src.engine import compile_utils
  import tf_keras as keras
else:
  from keras.src.engine import compile_utils

from tensorflow.keras.losses import BinaryCrossentropy
from deepray.losses.contrastive import contrastive_loss, ContrastiveLoss
from deepray.losses.giou_loss import giou_loss, GIoULoss
from deepray.losses.kappa_loss import WeightedKappaLoss
from deepray.losses.lifted import lifted_struct_loss, LiftedStructLoss
from deepray.losses.npairs import (
  npairs_loss,
  NpairsLoss,
  npairs_multilabel_loss,
  NpairsMultilabelLoss,
)
from deepray.losses.quantiles import pinball_loss, PinballLoss
from deepray.losses.sparsemax_loss import sparsemax_loss, SparsemaxLoss
from deepray.losses.triplet import (
  triplet_semihard_loss,
  triplet_hard_loss,
  TripletSemiHardLoss,
  TripletHardLoss,
)
from deepray.losses.softmax_loss import SoftmaxLoss


class Loss(compile_utils.LossesContainer):
  """Abstract class for all metrics."""

  def __init__(self, losses=BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)):
    super().__init__(losses)

  @abc.abstractmethod
  def call(self, y_true, y_pred, sample_weight=None):
    """
    must defined in subclass
    """
    raise NotImplementedError("call: not implemented!")

  def __call__(self, y_true, y_pred, sample_weight=None, regularization_losses=None):
    if not self._built:
      self._built = True
    loss_value = self.call(y_true, y_pred, sample_weight)
    total_loss_mean_value = tf.nn.compute_average_loss(
      loss_value, global_batch_size=flags.FLAGS.batch_size * flags.FLAGS.num_accumulation_steps
    )

    self._loss_metric.update_state(
      total_loss_mean_value,
      # sample_weight=batch_dim
    )
    return total_loss_mean_value

  def __repr__(self):
    return str(self)

  def __str__(self):
    return self.__class__.__name__
