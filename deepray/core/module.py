import time
from collections.abc import Iterator

import tensorflow as tf
from absl import logging, flags
from packaging import version

from deepray.core.common import distribution_utils
from deepray.utils import export
from deepray.utils.horovod_utils import is_main_process

FLAGS = flags.FLAGS


class Module():

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.eval_steps = None

  def steps_to_run(self, current_step, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
      raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
      return steps_per_loop, FLAGS.num_accumulation_steps

    # Note: broadcast should be called after the first gradient step to ensure optimizer
    # initialization.
    # if self.use_horovod and self.current_step == self._first_steps:
    #   return 1, 1

    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
      return min(steps_per_epoch - remainder_in_epoch, steps_per_loop), FLAGS.num_accumulation_steps
    else:
      return steps_per_loop, FLAGS.num_accumulation_steps

  def _float_metric_value(self, metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)

  def on_epoch_begin(self, epoch):
    self._step_epoch = 0
    """Calls the `on_epoch_begin` methods of its callbacks.
    """
    self.callbacks.on_epoch_begin(epoch)

    # Training loss/metric are taking average over steps inside micro
    # training loop. We reset their values before each round.
    self.loss_container.reset_state()
    if self.metric_container:
      self.metric_container.reset_state()

  def on_batch_end(self, logs, steps, t0):
    """Runs custom callbacks at the end of every N(steps) step."""
    self._step_epoch += steps
    self.current_step += steps

    self.callbacks.on_train_batch_end(self.current_step, logs)

    elapse_time = time.time() - t0
    # Updates training logging.
    if self.steps_per_epoch > 0:
      training_status = 'Train Step: %d/%d / time=%.3f sec' % (
          self.current_step, self.steps_per_epoch * self.epochs + self._first_steps, elapse_time
      )
    else:
      training_status = 'Train Step: %d / time=%.3f sec' % (self.current_step, elapse_time)

    self._steps_from_save += steps

    if is_main_process() and self._steps_from_save >= FLAGS.save_checkpoint_steps:
      export.export_to_checkpoint(self.manager, self.current_step)
      self._steps_from_save = 0

    if self.train_summary_writer:
      with self.train_summary_writer.as_default():
        for metric in self.metrics:
          metric_value = self._float_metric_value(metric)
          training_status += '  %s=%f' % (metric.name, metric_value)
          tf.summary.scalar(metric.name, metric_value, step=self.current_step)
        self.train_summary_writer.flush()

    # The number of samples trained per second
    step_throughput = self._performance_calculator(steps, self.global_batch_size)
    if is_main_process():
      if self.use_float16:
        if version.parse(tf.keras.__version__.replace("-tf", "+tf")) < version.parse("2.11"):
          logging.info(
              'Step: %d Lr %g Loss scale %g' %
              (self.current_step, self.optimizer._optimizer._decayed_lr('float32'), self.optimizer.loss_scale)
          )
        else:
          logging.info(
              'Step: %d Lr %g Loss scale %g' % (self.current_step, self.optimizer.lr, self.optimizer.loss_scale)
          )

      logging.info(training_status)
      logging.info('Perf %.2f samples/s' % step_throughput)

    if self.current_step > self._first_steps + steps * 2:
      self._perf_wo += step_throughput
      self._perf_wo_n += 1

  def on_epoch_end(self, epoch, current_step, eval_input, epoch_logs=None):
    # Saves model checkpoints and run validation steps at every epoch end.
    # To avoid repeated model saving, we do not save after the last step of training.
    if epoch < self.epochs - 1:
      export.export_to_checkpoint(self.manager, current_step)
    if eval_input:  # and is_main_process():
      if is_main_process():
        logging.info('Running evaluation after step: %s.', current_step)
      # Re-initialize evaluation metric.
      self.loss_container.reset_state()
      if self.metric_container:
        self.metric_container.reset_state()

      val_logs = self.evaluate(eval_input, self.eval_steps)
      val_logs = {'val_' + name: val for name, val in val_logs.items()}
      epoch_logs.update(val_logs)

      if is_main_process():
        with self.eval_summary_writer.as_default():
          for name, value in val_logs.items():
            logging.info('Step: [%d] Validation %s = %f', current_step, name, value)
            tf.summary.scalar(name, value, step=current_step)
          self.eval_summary_writer.flush()
    """Calls the `on_epoch_end` methods of its callbacks.
    """
    self.callbacks.on_epoch_end(epoch, epoch_logs)

  def evaluate(self, eval_input: tf.data.Dataset, eval_steps: int = None):
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        eval_input: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely).
          If `x` is a dataset, generator or `keras.utils.Sequence` instance,
          `y` should not be specified (since targets will be obtained from
          the iterator/dataset).
        eval_steps: Integer or `None`. Total number of steps (batches of samples)
          before declaring the evaluation round finished. Ignored with the
          default value of `None`. If x is a `tf.data` dataset and `steps`
          is None, 'evaluate' will run until the dataset is exhausted. This
          argument is not supported with array inputs.


    See the discussion of `Unpacking behavior for iterator-like inputs` for
    `Model.fit`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: If `trainer.evaluate` is wrapped in a `tf.function`.
    """

    if eval_steps is None:
      if self.eval_steps is not None:
        eval_steps = self.eval_steps
    else:

      if self.eval_steps is None:
        self.eval_steps = eval_steps
      """Runs validation steps and aggregate metrics."""
      if self.eval_steps is None:
        self.eval_steps = eval_steps

    if not isinstance(eval_input, Iterator):
      eval_input = distribution_utils.make_distributed_iterator(self.strategy, eval_input)

    current_step = 0
    while eval_steps is None or current_step < eval_steps:
      try:
        t0 = time.time()
        steps, _ = self.steps_to_run(current_step, eval_steps, FLAGS.steps_per_summary)
        for _ in tf.range(steps):
          self.forward_step(next(eval_input))
          current_step += 1
        elapse_time = time.time() - t0
        # Updates validing logging.
        if eval_steps is None:
          training_status = 'Valid Step: %d / time=%.3f sec' % (current_step, elapse_time)
        else:
          training_status = 'Valid Step: %d/%d / time=%.3f sec' % (current_step, eval_steps, elapse_time)
        for key, value in self.get_metrics_result().items():
          metric_value = value.numpy().astype(float)
          training_status += '  %s=%f' % (key, metric_value)
        if is_main_process():
          logging.info(training_status)
      except (tf.errors.OutOfRangeError, StopIteration):
        self.eval_steps = current_step
        if is_main_process():
          logging.info('Data exhausted after %d eval_steps', current_step)
        break

    return self.get_metrics_result()

  @property
  def metrics(self):
    metrics = []
    if self.loss_container is not None:
      metrics += self.loss_container.metrics
    if self.metric_container is not None:
      metrics += self.metric_container.metrics
    return metrics

  def get_metrics_result(self):
    """Returns the model's metrics values as a dict.

    If any of the metric result is a dict (containing multiple metrics),
    each of them gets added to the top level returned dict of this method.

    Returns:
      A `dict` containing values of the metrics listed in `self.metrics`.
      Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics
