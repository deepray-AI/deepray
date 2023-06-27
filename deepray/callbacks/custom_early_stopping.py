import tensorflow as tf
from absl import logging


class CustomEarlyStopping(tf.keras.callbacks.Callback):
  """Stop training has reached a desired hit rate."""

  def __init__(self, monitor, desired_value):
    super(CustomEarlyStopping, self).__init__()

    self.monitor = monitor
    self.desired = desired_value
    self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current and current >= self.desired:
      self.stopped_epoch = epoch
      self.model.stop_training = True

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning(
        "Early stopping conditioned on metric `%s` "
        "which is not available. Available metrics are: %s", self.monitor,
        ",".join(list(logs.keys())))
    return monitor_value
