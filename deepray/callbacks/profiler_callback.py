from tf_keras.callbacks import Callback
from tensorflow.python.eager import profiler

from deepray.utils import logging_util

logger = logging_util.get_logger()


def get_profiler_callback(model_dir, profile_steps, enable_tensorboard, steps_per_epoch):
  """Validate profile_steps flag value and return profiler callback."""
  profile_steps_error_message = (
    "profile_steps must be a comma separated pair of positive integers, "
    "specifying the first and last steps to be profiled."
  )
  try:
    profile_steps = [int(i) for i in profile_steps.split(",")]
  except ValueError:
    raise ValueError(profile_steps_error_message)
  if len(profile_steps) != 2:
    raise ValueError(profile_steps_error_message)
  start_step, stop_step = profile_steps
  if start_step < 0 or start_step > stop_step:
    raise ValueError(profile_steps_error_message)
  if enable_tensorboard:
    logger.warning(
      "Both TensorBoard and profiler callbacks are used. Note that the "
      "TensorBoard callback profiles the 2nd step (unless otherwise "
      "specified). Please make sure the steps profiled by the two callbacks "
      "do not overlap."
    )
  return ProfilerCallback(model_dir, start_step, stop_step, steps_per_epoch)


class ProfilerCallback(Callback):
  """Save profiles in specified step range to log directory."""

  def __init__(self, log_dir, start_step, stop_step, steps_per_epoch):
    super(ProfilerCallback, self).__init__()
    self.log_dir = log_dir
    self.start_step = start_step
    self.stop_step = stop_step
    self.start_epoch = start_step // steps_per_epoch
    self.stop_epoch = stop_step // steps_per_epoch
    self.start_step_in_epoch = start_step % steps_per_epoch
    self.stop_step_in_epoch = stop_step % steps_per_epoch
    self.should_start = False
    self.should_stop = False

  def on_epoch_begin(self, epoch, logs=None):
    if epoch == self.start_epoch:
      self.should_start = True
    if epoch == self.stop_epoch:
      self.should_stop = True

  def on_batch_begin(self, batch, logs=None):
    if batch == self.start_step_in_epoch and self.should_start:
      self.should_start = False
      profiler.start()
      logger.info("Profiler started at Step %s", self.start_step)

  def on_batch_end(self, batch, logs=None):
    if batch == self.stop_step_in_epoch and self.should_stop:
      self.should_stop = False
      results = profiler.stop()
      profiler.save(self.log_dir, results)
      logger.info(
        "Profiler saved profiles for steps between %s and %s to %s", self.start_step, self.stop_step, self.log_dir
      )
