import os
import re
import time
from collections.abc import Iterator

import tensorflow as tf
from absl import logging, flags
from packaging import version

from deepray.core.common import distribution_utils

FLAGS = flags.FLAGS


class Module():

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

  def _save_checkpoint(self, manager, checkpoint_number=None):
    """Saves model to with provided checkpoint prefix."""
    latest_checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'ckpt'))
    match = re.search(r"(?<=ckpt-)\d+", latest_checkpoint_file) if latest_checkpoint_file else None
    latest_step_ckpt = int(match.group()) if match else -1

    if latest_step_ckpt != checkpoint_number:
      save_path = manager.save(checkpoint_number)
      logging.info('Saved checkpoint to {}'.format(save_path))

  def _float_metric_value(self, metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)

  def on_train_begin(self):
    """Calls the `on_train_begin` methods of its callbacks.
    """
    for callback in self.callbacks:
      callback.on_train_begin()

  def on_epoch_begin(self, epoch):
    self._step_epoch = 0
    """Calls the `on_epoch_begin` methods of its callbacks.
    """
    for callback in self.callbacks:
      callback.on_epoch_begin(epoch)

    # Training loss/metric are taking average over steps inside micro
    # training loop. We reset their values before each round.
    self.loss_container.reset_state()
    if self.metric_container:
      self.metric_container.reset_state()

  def on_batch_begin(self, batch):
    """Runs custom callbacks at the start of every step."""
    for callback in self.callbacks:
      callback.on_train_batch_begin(batch)

  def on_batch_end(self, logs, steps, t0):
    """Runs custom callbacks at the end of every step."""
    for callback in self.callbacks:
      callback.on_train_batch_end(self.current_step, logs)

    self._step_epoch += steps
    self.current_step += steps

    elapse_time = time.time() - t0
    # Updates training logging.
    if self.steps_per_epoch > 0:
      training_status = 'Train Step: %d/%d / time=%.3f sec' % (
          self.current_step, self.steps_per_epoch * self.epochs, elapse_time
      )
    else:
      training_status = 'Train Step: %d / time=%.3f sec' % (self.current_step, elapse_time)

    self._steps_from_save += steps
    if FLAGS.use_horovod:
      import horovod.tensorflow as hvd
    if (not self.use_horovod or hvd.rank() == 0) and self._steps_from_save >= FLAGS.save_checkpoint_steps:
      self._save_checkpoint(self.manager, self.current_step)
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
    if not self.use_horovod or hvd.rank() == 0:
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

  def on_epoch_end(self, epoch, current_step, eval_input):
    # Saves model checkpoints and run validation steps at every epoch end.
    # To avoid repeated model saving, we do not save after the last step of training.
    if FLAGS.use_horovod:
      import horovod.tensorflow as hvd
    if (epoch < self.epochs - 1) and (not self.use_horovod or hvd.rank() == 0):
      self._save_checkpoint(self.manager, current_step)
      if self.sub_model:
        self._save_checkpoint(self.sub_manager)
    if eval_input and (epoch < self.epochs - 1) and (not self.use_horovod or hvd.rank() == 0):
      logging.info('Running evaluation after step: %s.', current_step)
      self.run_evaluation(eval_input, current_step)
      # Re-initialize evaluation metric.
      if self.metric_container:
        self.metric_container.reset_state()
    """Calls the `on_epoch_end` methods of its callbacks.
    """
    for callback in self.callbacks:
      callback.on_epoch_end(epoch)

  def on_train_end(self):
    """Calls the `on_train_begin` methods of its callbacks.
    """
    for callback in self.callbacks:
      callback.on_train_end()

  def run_evaluation(self, eval_input, current_training_step=0):
    """Runs validation steps and aggregate metrics."""
    if not isinstance(eval_input, Iterator):
      eval_input = distribution_utils.make_distributed_iterator(self.strategy, eval_input)

    step_num = 0
    while 1:
      try:
        self._test_step(eval_input)
        step_num += 1
      except (tf.errors.OutOfRangeError, StopIteration):
        logging.info('Data exhausted after %d steps', step_num)
        with self.eval_summary_writer.as_default():
          for metric in self.metrics:
            metric_value = self._float_metric_value(metric)
            logging.info('Step: [%d] Validation %s = %f', current_training_step, metric.name, metric_value)
            tf.summary.scalar(metric.name, metric_value, step=current_training_step)
          self.eval_summary_writer.flush()
        break

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

  def save_spec(self):
    # tf 2.6 以上版本
    # harbor.weizhipin.com/arsenal_notebook/tensorflow26:test
    if hasattr(self.model, 'save_spec'):
      return self.model.save_spec()
    else:
      arg_specs = list()
      kwarg_specs = dict()
      for i in self.model.inputs:
        arg_specs.append(i.type_spec)
      return [arg_specs], kwarg_specs

  def save_model_to_export(self, epoch=None):
    before = time.time()
    model_save_dir = os.path.join(FLAGS.model_dir, 'export')
    os.makedirs(model_save_dir, exist_ok=True)

    logging.info(f"save pb model to:{model_save_dir}, without optimizer & traces")

    @tf.function
    def serve(*args, **kwargs):
      return self.model(*args, **kwargs)

    arg_specs, kwarg_specs = self.save_spec()
    if FLAGS.use_dynamic_embedding:
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      tf.keras.models.save_model(
          self.model,
          model_save_dir,
          overwrite=True,
          include_optimizer=False,
          save_traces=False,
          options=options,
          signatures={'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)}
      )
    else:
      tf.keras.models.save_model(
          self.model,
          model_save_dir,
          overwrite=True,
          include_optimizer=False,
          save_traces=False,
          signatures={'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)}
      )

    after = time.time()
    logging.info(f"save pb model done at: {model_save_dir}. spend {after - before:.3f}s")

  def save_model_to_pb(self, epoch=None):
    before = time.time()
    model_save_dir = os.path.join(FLAGS.model_dir, 'pb') + "/model%s/" % ("" if epoch is None else "_%d" % epoch)
    os.makedirs(model_save_dir, exist_ok=True)
    logging.info(f"save pb model to:{model_save_dir}")
    options = None
    if FLAGS.use_dynamic_embedding:
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    tf.saved_model.save(self.model, export_dir=model_save_dir, options=options)
    after = time.time()
    logging.info(f"save pb model done at: {model_save_dir}. spend {after - before:.3f}s")

  def save_model_to_serving(self):  # 4 zero init
    export_model_path = os.path.join(FLAGS.model_dir, 'export_tfra')
    path_to_variable = os.path.join(export_model_path, "variables", "variables")
    logging.info(f"zero init, load export model from: {path_to_variable}")

    tf.keras.backend.clear_session()
    zero_model = self.build_model(is_training=False)
    zero_model.load_weights(path_to_variable)
    logging.info("zero init, load export model done and start save model.")

    zero_model_save_path = os.path.join(FLAGS.model_dir, 'export_zero')
    os.makedirs(zero_model_save_path, exist_ok=True)

    @tf.function
    def serve(*args, **kwargs):
      return zero_model(*args, **kwargs)

    arg_specs, kwarg_specs = self.save_spec_self(zero_model)
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    tf.keras.models.save_model(
        zero_model,
        zero_model_save_path,
        overwrite=True,
        include_optimizer=False,
        save_traces=False,
        options=options,
        signatures={'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)}
    )
    logging.info("zero init, save model done.")

  def save_spec_self(self, model):
    # tf 2.6 以上版本
    # harbor.weizhipin.com/arsenal_notebook/tensorflow26:test
    if hasattr(model, 'save_spec'):
      return model.save_spec()
    else:
      arg_specs = list()
      kwarg_specs = dict()
      for i in model.inputs:
        arg_specs.append(i.type_spec)
      return [arg_specs], kwarg_specs