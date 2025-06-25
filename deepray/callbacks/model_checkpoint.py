# Copyright 2023 The Deepray Authors. All Rights Reserved.
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
import os
import sys

import tensorflow as tf
from absl import flags
from tf_keras.callbacks import Callback
from typeguard import typechecked

from deepray.utils import export
from deepray.utils import logging_util
from deepray.utils.horovod_utils import is_main_process, get_world_size, get_rank

logger = logging_util.get_logger()


@tf.keras.utils.register_keras_serializable(package="Deepray")
class ModelCheckpoint(Callback):
  @typechecked
  def __init__(self, save_checkpoint_steps: int = sys.maxsize, max_to_keep: int = 3):
    super().__init__()
    self.save_checkpoint_steps = save_checkpoint_steps
    self.max_to_keep = max_to_keep
    self.epochs = flags.FLAGS.epochs
    if flags.FLAGS.stop_steps >= 0:
      self.epochs = 1
    if flags.FLAGS.use_dynamic_embedding:
      from tensorflow_recommenders_addons import dynamic_embedding as de

      tf.train.Checkpoint = de.train.checkpoint.DECheckpoint

  def set_models(self, models):
    self.models = models

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  # def set_iterator(self, iterator):
  #   self.iterator = iterator

  @property
  def manager(self):
    if len(self._managers) == 1:
      return self._managers["main"]
    else:
      return self._managers

  def on_callback_begin(self):
    self._checkpoints, self._managers = {}, {}
    for name, model in self.models.items():
      if "main" in name:
        _checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        self._checkpoints[name] = _checkpoint
        if get_world_size() > 1:
          self._managers[name] = tf.train.CheckpointManager(
            _checkpoint, os.path.join(flags.FLAGS.model_dir, f"ckpt_{name}_{get_rank()}"), max_to_keep=self.max_to_keep
          )
        else:
          self._managers[name] = tf.train.CheckpointManager(
            _checkpoint, os.path.join(flags.FLAGS.model_dir, f"ckpt_{name}"), max_to_keep=self.max_to_keep
          )
      else:
        _checkpoint = tf.train.Checkpoint(model=model)
        self._checkpoints[name] = _checkpoint
        self._managers[name] = tf.train.CheckpointManager(
          _checkpoint, os.path.join(flags.FLAGS.model_dir, f"ckpt_{name}"), max_to_keep=self.max_to_keep
        )

    if flags.FLAGS.init_checkpoint:
      for (name, ckpt), init_ckpt in zip(self._checkpoints.items(), flags.FLAGS.init_checkpoint):
        if init_ckpt:
          if tf.io.gfile.isdir(init_ckpt):
            latest_checkpoint = tf.train.latest_checkpoint(init_ckpt)
          else:
            latest_checkpoint = init_ckpt
          logger.info(
            f"Checkpoint file {latest_checkpoint} found and restoring from initial checkpoint for {name} model."
          )
          if os.getenv("DEEPRAY_VERBOSITY", None) == "detail" or flags.FLAGS.use_dynamic_embedding:
            # TFRA DE doesn't support "assert_existing_objects_matched" method
            ckpt.restore(latest_checkpoint)
          else:
            ckpt.restore(latest_checkpoint).assert_existing_objects_matched()
          logger.info("Loading from checkpoint file...")

    self.current_step = 0
    self._steps_from_save = 0  # self.optimizer.iterations.numpy()

  def on_train_begin(self, logs=None):
    self.on_callback_begin()

  def on_test_begin(self, logs=None):
    self.on_callback_begin()

  def on_predict_begin(self, logs=None):
    self.on_callback_begin()

  def on_train_batch_end(self, batch, logs=None):
    self.current_step = batch
    if self._steps_from_save + self.save_checkpoint_steps <= batch:
      export.export_to_checkpoint(self.manager, batch)
      self._steps_from_save = batch

  def on_epoch_end(self, epoch, logs=None):
    # Saves model checkpoints and run validation steps at every epoch end.
    # To avoid repeated model saving, we do not save after the last step of training.
    if epoch < self.epochs - 1:
      export.export_to_checkpoint(self.manager, self.current_step)

  def on_train_end(self, logs=None):
    export.export_to_checkpoint(self.manager, self.current_step)

  def get_config(self):
    config = {
      "save_checkpoint_steps": self.save_checkpoint_steps,
      "max_to_keep": self.max_to_keep,
    }

    base_config = super().get_config()
    return {**base_config, **config}


class SimpleCheckpoint(Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)
