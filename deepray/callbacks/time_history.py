import time

import tensorflow as tf
from tf_keras.callbacks import Callback
from tensorflow.python.eager import monitoring

from deepray.utils import logging_util

logger = logging_util.get_logger()

global_batch_size_gauge = monitoring.IntGauge("/tensorflow/training/global_batch_size", "TF training global batch size")
first_batch_time_gauge = monitoring.IntGauge(
  "/tensorflow/training/first_batch", "TF training start/end time for first batch (unix epoch time in us.", "type"
)

first_batch_start_time = first_batch_time_gauge.get_cell("start")
first_batch_end_time = first_batch_time_gauge.get_cell("end")


class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(self.batch_index, self.timestamp)


class TimeHistory(Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, initial_step=0, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      initial_step: Optional, initial step.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = initial_step
    self.steps_before_epoch = initial_step
    self.steps_in_epoch = 0
    self.start_time = None

    global_batch_size_gauge.get_cell().set(batch_size)

    if logdir:
      self.summary_writer = tf.summary.create_file_writer(logdir)
    else:
      self.summary_writer = None

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return self.global_steps / sum(self.epoch_runtime_log)

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    return self.average_steps_per_second * self.batch_size

  def get_examples_per_sec(self, warmup=1):
    """Calculates examples/sec through timestamp_log and skip warmup period."""
    # First entry in timestamp_log is the start of the step 1. The rest of the
    # entries are the end of each step recorded.
    time_log = self.timestamp_log
    seconds = time_log[-1].timestamp - time_log[warmup].timestamp
    steps = time_log[-1].batch_index - time_log[warmup].batch_index
    return self.batch_size * steps / seconds

  def get_startup_time(self, start_time_sec):
    return self.timestamp_log[0].timestamp - start_time_sec

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

    if self.summary_writer:
      self.summary_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()
      if not first_batch_start_time.value():
        first_batch_start_time.set(int(self.start_time * 1000000))

    # Record the timestamp of the first global step
    if not self.timestamp_log:
      self.timestamp_log.append(BatchTimestamp(self.global_steps, self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if not first_batch_end_time.value():
      first_batch_end_time.set(int(time.time() * 1000000))
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / elapsed_time
      examples_per_second = steps_per_second * self.batch_size

      self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
      logger.info(
        "TimeHistory: %.2f seconds, %.2f examples/second between steps %d and %d",
        elapsed_time,
        examples_per_second,
        self.last_log_step,
        self.global_steps,
      )

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar("steps_per_second", steps_per_second, self.global_steps)
          tf.summary.scalar("examples_per_second", examples_per_second, self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0
