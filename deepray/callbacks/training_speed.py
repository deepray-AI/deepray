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

import tensorflow as tf
from absl import flags
from tf_keras.callbacks import Callback
from tf_keras.src.utils import io_utils

from deepray.utils import logging_util
from deepray.utils.benchmark import PerformanceCalculator
from deepray.utils.horovod_utils import get_world_size, is_main_process

logger = logging_util.get_logger()


class TrainingSpeed(Callback):
  """Callback that prints metrics to stdout.

  Args:
      count_mode: One of `"steps"` or `"samples"`.
          Whether the progress bar should
          count samples seen or steps (batches) seen.

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, batch_size: int = None):
    super().__init__()
    local_batch_size = batch_size or flags.FLAGS.batch_size
    logger.info(f"Callback using local (per-replica) batch_size: {local_batch_size}")

    if flags.FLAGS.use_horovod:
      world_size = get_world_size()
      self.global_batch_size = local_batch_size * world_size
      if is_main_process():
        logger.info(f"Horovod enabled: global_batch_size set to {self.global_batch_size} ({world_size} workers)")
    else:
      self.global_batch_size = local_batch_size

    self.seen = 0
    self.performance_calculator = None
    self.epochs = 1

    self._train_step, self._test_step, self._predict_step = None, None, None
    self._call_batch_hooks = True

    self._called_in_fit = False

  def set_params(self, params):
    self.epochs = params["epochs"]
    self._call_batch_hooks = True
    try:
      self._train_step = self.model._train_counter
      self._test_step = self.model._test_counter
      self._predict_step = self.model._predict_counter
    except AttributeError:
      self._call_batch_hooks = True

    self.last_step = 0
    if isinstance(self.last_step, (tf.Tensor, tf.Variable)):
      self.last_step = self.last_step.numpy()

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  def on_train_begin(self, logs=None):
    # When this logger is called inside `fit`, validation is silent.
    self._called_in_fit = True
    self._perf_wo = 0
    self._perf_wo_n = 0

    # Training loop starts here.
    if hasattr(self.optimizer, "iterations"):
      self._first_steps = self.optimizer.iterations.numpy()
    else:
      self._first_steps = 0

  def on_test_begin(self, logs=None):
    if not self._called_in_fit:
      self._reset_progbar()
      self._maybe_init_progbar()

  def on_predict_begin(self, logs=None):
    self._reset_progbar()
    self._maybe_init_progbar()

  def on_train_batch_end(self, batch, logs=None):
    if is_main_process():
      self._batch_update_progbar(batch, logs)

  def on_test_batch_end(self, batch, logs=None):
    if not self._called_in_fit:
      self._batch_update_progbar(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    # Don't pass prediction results.
    self._batch_update_progbar(batch, None)

  def on_test_end(self, logs=None):
    if not self._called_in_fit:
      self._finalize_progbar(logs, self._test_step)

  def on_predict_end(self, logs=None):
    self._finalize_progbar(logs, self._predict_step)

  def _reset_progbar(self):
    self.seen = 0
    self.performance_calculator = None

  def _maybe_init_progbar(self):
    if self.performance_calculator is None:
      self.performance_calculator = PerformanceCalculator()

  def _implements_train_batch_hooks(self):
    return self._call_batch_hooks

  def _implements_test_batch_hooks(self):
    return self._call_batch_hooks

  def _implements_predict_batch_hooks(self):
    return self._call_batch_hooks

  def _batch_update_progbar(self, batch, logs=None):
    """Updates the performance_calculator."""
    self._maybe_init_progbar()
    self.seen = batch + 1  # One-indexed.
    delta_steps = self.seen - self.last_step

    step_throughput = self.performance_calculator(delta_steps, self.global_batch_size)
    logger.info("Perf %.2f samples/s" % step_throughput)

    if batch > self._first_steps + delta_steps * 2:
      self._perf_wo += step_throughput
      self._perf_wo_n += 1

    self.last_step = self.seen

  def _finalize_progbar(self, logs, counter):
    results_perf = self.performance_calculator.get_current_benchmark_results()
    logger.info(results_perf)
    if self._perf_wo_n != 0:
      logger.info("Throughput Average (examples/sec) = %0.2f", self._perf_wo / self._perf_wo_n)
