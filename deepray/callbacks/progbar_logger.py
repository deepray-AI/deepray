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
import json
import os
import time
import copy
import numpy as np
import sys

import tensorflow as tf
from absl import flags
from tf_keras.callbacks import Callback
from tf_keras.src.utils import io_utils
from tf_keras.src.utils import tf_utils

from deepray.utils import logging_util
from deepray.utils.benchmark import PerformanceCalculator
from deepray.utils.flags import common_flags
from deepray.utils.horovod_utils import is_main_process, get_world_size

logger = logging_util.get_logger()


class Progbar:
  """Displays a progress bar.

  Args:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that should *not*
        be averaged over time. Metrics in this list will be displayed as-is.
        All others will be averaged by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually "step" or "sample").
  """

  def __init__(
    self,
    target,
    width=30,
    verbose=1,
    interval=0.05,
    stateful_metrics=None,
    unit_name="step",
  ):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    self.unit_name = unit_name
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = (
      (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
      or "ipykernel" in sys.modules
      or "posix" in sys.modules
      or "PYCHARM_HOSTED" in os.environ
    )
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0
    self._time_at_epoch_start = self._start
    self._time_at_epoch_end = None
    self._time_after_first_step = None

  def update(self, current, values=None, finalize=None):
    """Updates the progress bar.

    Args:
        current: Index of current step.
        values: List of tuples: `(name, value_for_last_step)`. If `name` is
          in `stateful_metrics`, `value_for_last_step` will be displayed
          as-is. Else, an average of the metric over time will be
          displayed.
        finalize: Whether this is the last update for the progress bar. If
          `None`, uses `current >= self.target`. Defaults to `None`.
    """
    if finalize is None:
      if self.target is None:
        finalize = False
      else:
        finalize = current >= self.target

    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        # In the case that progress bar doesn't have a target value in
        # the first epoch, both on_batch_end and on_epoch_end will be
        # called, which will cause 'current' and 'self._seen_so_far' to
        # have the same value. Force the minimal value to 1 here,
        # otherwise stateful_metric will be 0s.
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    message = ""
    now = time.time()
    info = f" - {now - self._start:.0f}s"
    if current == self.target:
      self._time_at_epoch_end = now
    if self.verbose == 1:
      if now - self._last_update < self.interval and not finalize:
        return

      prev_total_width = self._total_width
      if self._dynamic_display:
        message += "\b" * prev_total_width
        message += "\r"
      else:
        message += "\n"

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += "=" * (prog_width - 1)
          if current < self.target:
            bar += ">"
          else:
            bar += "="
        bar += "." * (self.width - prog_width)
        bar += "]"
      else:
        bar = "%7d/Unknown" % current

      self._total_width = len(bar)
      message += bar

      time_per_unit = self._estimate_step_duration(current, now)

      if self.target is None or finalize:
        info += self._format_time(time_per_unit, self.unit_name)
      else:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = "%d:%02d:%02d" % (
            eta // 3600,
            (eta % 3600) // 60,
            eta % 60,
          )
        elif eta > 60:
          eta_format = "%d:%02d" % (eta // 60, eta % 60)
        else:
          eta_format = "%ds" % eta

        info = f" - ETA: {eta_format}"

      for k in self._values_order:
        info += f" - {k}:"
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += f" {avg:.4f}"
          else:
            info += f" {avg:.4e}"
        else:
          info += f" {self._values[k]}"

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += " " * (prev_total_width - self._total_width)

      if finalize:
        info += "\n"

      message += info
      logger.info(message)
      # io_utils.print_msg(message, line_break=False)
      message = ""

    elif self.verbose == 2:
      if finalize:
        numdigits = int(np.log10(self.target)) + 1
        count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += f" - {k}:"
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += f" {avg:.4f}"
          else:
            info += f" {avg:.4e}"
        if self._time_at_epoch_end:
          time_per_epoch = self._time_at_epoch_end - self._time_at_epoch_start
          avg_time_per_step = time_per_epoch / self.target
          self._time_at_epoch_start = now
          self._time_at_epoch_end = None
          info += " -" + self._format_time(time_per_epoch, "epoch")
          info += " -" + self._format_time(avg_time_per_step, self.unit_name)
          info += "\n"
        message += info
        io_utils.print_msg(message, line_break=False)
        message = ""

    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)

  def _format_time(self, time_per_unit, unit_name):
    """format a given duration to display to the user.

    Given the duration, this function formats it in either milliseconds
    or seconds and displays the unit (i.e. ms/step or s/epoch)
    Args:
      time_per_unit: the duration to display
      unit_name: the name of the unit to display
    Returns:
      a string with the correctly formatted duration and units
    """
    formatted = ""
    if time_per_unit >= 1 or time_per_unit == 0:
      formatted += f" {time_per_unit:.0f}s/{unit_name}"
    elif time_per_unit >= 1e-3:
      formatted += f" {time_per_unit * 1000.0:.0f}ms/{unit_name}"
    else:
      formatted += f" {time_per_unit * 1000000.0:.0f}us/{unit_name}"
    return formatted

  def _estimate_step_duration(self, current, now):
    """Estimate the duration of a single step.

    Given the step number `current` and the corresponding time `now` this
    function returns an estimate for how long a single step takes. If this
    is called before one step has been completed (i.e. `current == 0`) then
    zero is given as an estimate. The duration estimate ignores the duration
    of the (assumed to be non-representative) first step for estimates when
    more steps are available (i.e. `current>1`).

    Args:
      current: Index of current step.
      now: The current time.

    Returns: Estimate of the duration of a single step.
    """
    if current:
      # there are a few special scenarios here:
      # 1) somebody is calling the progress bar without ever supplying
      #    step 1
      # 2) somebody is calling the progress bar and supplies step one
      #    multiple times, e.g. as part of a finalizing call
      # in these cases, we just fall back to the simple calculation
      if self._time_after_first_step is not None and current > 1:
        time_per_unit = (now - self._time_after_first_step) / (current - 1)
      else:
        time_per_unit = (now - self._start) / current

      if current == 1:
        self._time_after_first_step = now
      return time_per_unit
    else:
      return 0

  def _update_stateful_metrics(self, stateful_metrics):
    self.stateful_metrics = self.stateful_metrics.union(stateful_metrics)


class ProgbarLogger(Callback):
  """Callback that prints metrics to stdout.

  Args:
      count_mode: One of `"steps"` or `"samples"`.
          Whether the progress bar should
          count samples seen or steps (batches) seen.
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is.
          All others will be averaged over time (e.g. loss, etc).
          If not provided, defaults to the `Model`'s metrics.

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, count_mode: str = "samples", stateful_metrics=None):
    super().__init__()
    self._supports_tf_logs = True
    if count_mode == "samples":
      self.use_steps = False
    elif count_mode == "steps":
      self.use_steps = True
    else:
      raise ValueError(f'Unknown `count_mode`: {count_mode}. Expected values are ["samples", "steps"]')
    # Defaults to all Model's metrics except for loss.
    self.stateful_metrics = set(stateful_metrics) if stateful_metrics else set()

    self.seen = 0
    self.progbar = None
    self.target = None
    self.verbose = 1
    self.epochs = 1

    self._train_step, self._test_step, self._predict_step = None, None, None
    self._call_batch_hooks = True

    self._called_in_fit = False

  def set_params(self, params):
    self.verbose = params["verbose"]
    self.epochs = params["epochs"]
    if self.use_steps and "steps" in params:
      self.target = params["steps"]
    elif not self.use_steps and "samples" in params:
      self.target = params["samples"]
    else:
      self.target = None  # Will be inferred at the end of the first epoch.

    self._call_batch_hooks = self.verbose == 1
    if self.target is None:
      try:
        self._train_step = self.model._train_counter
        self._test_step = self.model._test_counter
        self._predict_step = self.model._predict_counter
      except AttributeError:
        self._call_batch_hooks = True

  def on_train_begin(self, logs=None):
    # When this logger is called inside `fit`, validation is silent.
    self._called_in_fit = True

  def on_test_begin(self, logs=None):
    if not self._called_in_fit:
      self._reset_progbar()
      self._maybe_init_progbar()

  def on_predict_begin(self, logs=None):
    self._reset_progbar()
    self._maybe_init_progbar()

  def on_epoch_begin(self, epoch, logs=None):
    self._reset_progbar()
    self._maybe_init_progbar()
    if self.verbose and self.epochs > 1:
      io_utils.print_msg(f"Epoch {epoch + 1}/{self.epochs}")

  def on_train_batch_end(self, batch, logs=None):
    self._batch_update_progbar(batch, logs)

  def on_test_batch_end(self, batch, logs=None):
    if not self._called_in_fit:
      self._batch_update_progbar(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    # Don't pass prediction results.
    self._batch_update_progbar(batch, None)

  def on_epoch_end(self, epoch, logs=None):
    self._finalize_progbar(logs, self._train_step)

  def on_test_end(self, logs=None):
    if not self._called_in_fit:
      self._finalize_progbar(logs, self._test_step)

  def on_predict_end(self, logs=None):
    self._finalize_progbar(logs, self._predict_step)

  def _reset_progbar(self):
    self.seen = 0
    self.progbar = None

  def _maybe_init_progbar(self):
    """Instantiate a `Progbar` if not yet, and update the stateful
    metrics."""
    # TODO(rchao): Legacy TF1 code path may use list for
    # `self.stateful_metrics`. Remove "cast to set" when TF1 support is
    # dropped.
    self.stateful_metrics = set(self.stateful_metrics)

    if self.model:
      # Update the existing stateful metrics as `self.model.metrics` may
      # contain updated metrics after `MetricsContainer` is built in the
      # first train step.
      self.stateful_metrics = self.stateful_metrics.union(set(m.name for m in self.model.metrics))

    if self.progbar is None:
      self.progbar = Progbar(
        target=self.target,
        verbose=self.verbose,
        stateful_metrics=self.stateful_metrics,
        unit_name="step" if self.use_steps else "sample",
      )

    self.progbar._update_stateful_metrics(self.stateful_metrics)

  def _implements_train_batch_hooks(self):
    return self._call_batch_hooks

  def _implements_test_batch_hooks(self):
    return self._call_batch_hooks

  def _implements_predict_batch_hooks(self):
    return self._call_batch_hooks

  def _batch_update_progbar(self, batch, logs=None):
    """Updates the progbar."""
    logs = logs or {}
    self._maybe_init_progbar()
    if self.use_steps:
      self.seen = batch + 1  # One-indexed.
    else:
      # v1 path only.
      logs = copy.copy(logs)
      batch_size = logs.pop("size", 0)
      num_steps = logs.pop("num_steps", 1)
      logs.pop("batch", None)
      add_seen = num_steps * batch_size
      self.seen += add_seen

    if self.verbose == 1:
      # Only block async when verbose = 1.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.progbar.update(self.seen, list(logs.items()), finalize=False)

  def _finalize_progbar(self, logs, counter):
    logs = tf_utils.sync_to_numpy_or_python_type(logs or {})
    if self.target is None:
      if counter is not None:
        counter = counter.numpy()
        if not self.use_steps:
          counter *= logs.get("size", 1)
      self.target = counter or self.seen
      self.progbar.target = self.target
    self.progbar.update(self.target, list(logs.items()), finalize=True)
