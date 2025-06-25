from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a warmup cosine decay schedule."""

  def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
    """
    Applies a cosine warmup schedule.

    Args:
        lr_start: The initial learning rate
        lr_max: The maximum learning rate to which lr should increase to in
            the warmup steps
        warmup_steps: The number of steps for which the model warms up
        total_steps: The total number of steps for the model training
    """

    super().__init__()
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.pi = tf.constant(np.pi)

  def __call__(self, step):
    # Check whether the total number of steps is larger than the warmup
    # steps. If not, then throw a value error.
    if self.total_steps < self.warmup_steps:
      raise ValueError(
        f"Total number of steps {self.total_steps} must be" + f"larger or equal to warmup steps {self.warmup_steps}."
      )

    # `cos_annealed_lr` is a graph that increases to 1 from the initial
    # step to the warmup step. After that this graph decays to -1 at the
    # final step mark.
    cos_annealed_lr = tf.cos(
      self.pi
      * (tf.cast(step, tf.float32) - self.warmup_steps)
      / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
    )

    # Shift the mean of the `cos_annealed_lr` graph to 1. Now the grpah goes
    # from 0 to 2. Normalize the graph with 0.5 so that now it goes from 0
    # to 1. With the normalized graph we scale it with `lr_max` such that
    # it goes from 0 to `lr_max`
    learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

    # Check whether warmup_steps is more than 0.
    if self.warmup_steps > 0:
      # Check whether lr_max is larger that lr_start. If not, throw a value
      # error.
      if self.lr_max < self.lr_start:
        raise ValueError(f"lr_start {self.lr_start} must be smaller or" + f"equal to lr_max {self.lr_max}.")

      # Calculate the slope with which the learning rate should increase
      # in the warumup schedule. The formula for slope is m = ((b-a)/steps)
      slope = (self.lr_max - self.lr_start) / self.warmup_steps

      # With the formula for a straight line (y = mx+c) build the warmup
      # schedule
      warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

      # When the current step is lesser that warmup steps, get the line
      # graph. When the current step is greater than the warmup steps, get
      # the scaled cos graph.
      learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)

    # When the current step is more that the total steps, return 0 else return
    # the calculated graph.
    return tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")


# Some code is taken from:
# https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization_tf.py#L30.
class WarmUpPolynomial(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  Applies a warmup schedule on a given learning rate decay schedule.
  Args:
      initial_learning_rate (`float`):
          The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
          of the warmup).
      decay_schedule_fn (`Callable`):
          The schedule function to apply after the warmup for the rest of training.
      warmup_steps (`int`):
          The number of steps for the warmup part of training.
      power (`float`, *optional*, defaults to 1):
          The power to use for the polynomial warmup (defaults is a linear warmup).
      name (`str`, *optional*):
          Optional name prefix for the returned tensors during the schedule.
  """

  def __init__(
    self,
    initial_learning_rate: float,
    decay_schedule_fn: Callable,
    warmup_steps: int,
    power: float = 1.0,
    name: str = None,
  ):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmUp") as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
      return tf.cond(
        global_step_float < warmup_steps_float,
        lambda: warmup_learning_rate,
        lambda: self.decay_schedule_fn(step - self.warmup_steps),
        name=name,
      )

  def get_config(self):
    return {
      "initial_learning_rate": self.initial_learning_rate,
      "decay_schedule_fn": self.decay_schedule_fn,
      "warmup_steps": self.warmup_steps,
      "power": self.power,
      "name": self.name,
    }
