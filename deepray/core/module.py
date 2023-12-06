import time

import numpy as np
import tensorflow as tf
from absl import logging, flags
from keras import callbacks as callbacks_module
from keras.engine import base_layer
from keras.engine import data_adapter
from keras.engine.data_adapter import _ClusterCoordinatorDataHandler, DataHandler
from keras.utils import tf_utils
from keras.utils import version_utils
from packaging import version
from tensorflow.python.eager import context

from deepray.core.common import distribution_utils
from deepray.utils import export
from deepray.utils.horovod_utils import is_main_process

FLAGS = flags.FLAGS


def _minimum_control_deps(outputs):
  """Returns the minimum control dependencies to ensure step succeeded."""
  if tf.executing_eagerly():
    return []  # Control dependencies not needed.
  outputs = tf.nest.flatten(outputs, expand_composites=True)
  for out in outputs:
    # Variables can't be control dependencies.
    if not isinstance(out, tf.Variable):
      return [out]  # Return first Tensor or Op from outputs.
  return []  # No viable Tensor or Op to use for control deps.


def flatten_metrics_in_order(logs, metrics_names):
  """Turns the `logs` dict into a list as per key order of `metrics_names`."""
  results = []
  for name in metrics_names:
    if name in logs:
      results.append(logs[name])
  for key in sorted(logs.keys()):
    if key not in metrics_names:
      results.append(logs[key])
  if len(results) == 1:
    return results[0]
  return results


class DataHandlerMOD(DataHandler):

  def _validate_data_handler(self):
    pass


def get_data_handler(*args, **kwargs):
  if getattr(kwargs["model"], "_cluster_coordinator", None):
    return _ClusterCoordinatorDataHandler(*args, **kwargs)
  return DataHandlerMOD(*args, **kwargs)


class Module():

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._distribution_strategy = distribution_utils.get_distribution_strategy()
    self._init_batch_counters()
    self.eval_steps = None
    self._cluster_coordinator = None

    self.test_function = None

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _init_batch_counters(self):
    # Untracked Variables, used to keep track of mini-batches seen in `fit`,
    # `evaluate`, and `predict`.
    agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
    self._train_counter = tf.Variable(0, dtype='int64', aggregation=agg)
    self._test_counter = tf.Variable(0, dtype='int64', aggregation=agg)
    self._predict_counter = tf.Variable(0, dtype='int64', aggregation=agg)

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _configure_steps_per_execution(self, steps_per_execution):
    self._steps_per_execution = tf.Variable(
        steps_per_execution, dtype='int64', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
    )

  @property
  def distribute_strategy(self):
    """The `tf.distribute.Strategy` this model was created under."""
    return self._distribution_strategy or tf.distribute.get_strategy()

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

    if self._steps_from_save >= FLAGS.save_checkpoint_steps:
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

  def evaluate(self, eval_input: tf.data.Dataset, eval_steps: int = None, callbacks=None, return_dict=True, **kwargs):
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

    base_layer.keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')

    use_cached_eval_dataset = kwargs.pop('_use_cached_eval_dataset', False)
    if kwargs:
      raise TypeError(f'Invalid keyword arguments: {list(kwargs.keys())}')

    # TODO(@fuhailin): custom ProgbarLogger fix bug when verbose = 2
    verbose = 1
    with distribution_utils.get_strategy_scope(self._distribution_strategy):
      # Use cached evaluation data only when it's called in `Model.fit`
      if use_cached_eval_dataset and getattr(self, '_eval_data_handler', None) is not None:
        data_handler = self._eval_data_handler
      else:
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = get_data_handler(
            x=eval_input,
            y=None,
            sample_weight=None,
            batch_size=FLAGS.batch_size,
            steps_per_epoch=self.eval_steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            model=self.main_model,
            steps_per_execution=self._steps_per_execution
        )

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self.main_model,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps
        )
      logs = {}
      self.test_function = self.make_test_function()
      self._test_counter.assign(0)
      callbacks.on_test_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        # Re-initialize evaluation metric.
        self.reset_metrics()
        while eval_steps is None or self._test_counter.numpy() < eval_steps:
          try:
            steps, _ = self.steps_to_run(
                self._test_counter.numpy(),
                steps_per_epoch=eval_steps if eval_steps else -1,
                steps_per_loop=FLAGS.steps_per_summary
            )
            with tf.profiler.experimental.Trace('test', step_num=self._test_counter.numpy(), _r=1):
              callbacks.on_test_batch_begin(self._test_counter.numpy())
              tmp_logs = self.test_function(iterator, tf.convert_to_tensor(steps, dtype=tf.int32))
              if data_handler.should_sync:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              callbacks.on_test_batch_end(self._test_counter.numpy(), logs)
          except (tf.errors.OutOfRangeError, StopIteration):
            callbacks.on_test_batch_end(self._test_counter.numpy(), logs)
            self.eval_steps = self._test_counter.numpy()
            if is_main_process():
              logging.info('Data exhausted after %d eval_steps', self._test_counter.numpy())
            break

      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      callbacks.on_test_end(logs=logs)

      if return_dict:
        return logs
      else:
        return flatten_metrics_in_order(logs, self.metrics_names)

  def make_test_function(self, force=False):
    """Creates a function that executes one step of evaluation.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.evaluate` and `Model.test_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.test_step`.

    This function is cached the first time `Model.evaluate` or
    `Model.test_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called. You can skip the cache and generate again the
    function with `force=True`.

    Args:
      force: Whether to regenerate the test function and skip the cached
        function if available.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_test_batch_end`.
    """
    if self.test_function is not None and not force:
      return self.test_function

    def step_function(trainer, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = self.test_step(data)
        # Ensure counter is updated only if `test_step` succeeds.
        with tf.control_dependencies(_minimum_control_deps(outputs)):
          trainer._test_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      if self._jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)

      data = next(iterator)
      outputs = run_step(data)
      return outputs

    # Special case if steps_per_execution is one.
    if self._steps_per_execution is None or self._steps_per_execution.numpy().item() == 1:

      def test_function(iterator):
        """Runs a test execution with a single step."""
        return step_function(self, iterator)

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)

      if self._cluster_coordinator:
        self.test_function = lambda it: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
            test_function, args=(it,))
      else:
        self.test_function = test_function

    # If we're using a coordinator, use the value of self._steps_per_execution
    # at the time the function is called/scheduled, and not when it is actually
    # executed.
    elif self._cluster_coordinator:

      def test_function(iterator, steps_per_execution):
        """Runs a test execution with multiple steps."""
        for _ in tf.range(steps_per_execution):
          outputs = step_function(self, iterator)
        return outputs

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)

      self.test_function = lambda it: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
          test_function,
          args=(it, self._steps_per_execution.value()))
    else:

      def test_function(iterator, steps):
        """Runs a test execution with multiple steps."""
        for _ in tf.range(steps):
          outputs = step_function(self, iterator)
        return outputs

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)
      self.test_function = test_function

    return self.test_function

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

  def test_step(self, data):
    """The logic for one evaluation step.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.make_test_function`.

    This function should contain the mathematical logic for one step of
    evaluation.
    This typically includes the forward pass, loss calculation, and metrics
    updates.

    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_test_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned.
    """
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    y_pred = self.main_model(x, training=False)
    # Updates stateful loss metrics.
    self.compute_loss(x, y, y_pred, sample_weight)
    return self.compute_metrics(x, y, y_pred, sample_weight)

  def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
    """Compute the total loss, validate it, and return it.

    Subclasses can optionally override this method to provide custom loss
    computation logic.

    Example:
    ```python
    class MyModel(tf.keras.Model):

      def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

      def compute_loss(self, x, y, y_pred, sample_weight):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
        loss += tf.add_n(self.losses)
        self.loss_tracker.update_state(loss)
        return loss

      def reset_metrics(self):
        self.loss_tracker.reset_states()

      @property
      def metrics(self):
        return [self.loss_tracker]

    tensors = tf.random.uniform((10, 10)), tf.random.uniform((10,))
    dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)

    inputs = tf.keras.layers.Input(shape=(10,), name='my_input')
    outputs = tf.keras.layers.Dense(10)(inputs)
    model = MyModel(inputs, outputs)
    model.add_loss(tf.reduce_sum(outputs))

    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer, loss='mse', steps_per_execution=10)
    model.fit(dataset, epochs=2, steps_per_epoch=10)
    print('My custom loss: ', model.loss_tracker.result().numpy())
    ```

    Args:
      x: Input data.
      y: Target data.
      y_pred: Predictions returned by the model (output of `model(x)`)
      sample_weight: Sample weights for weighting the loss function.

    Returns:
      The total loss as a `tf.Tensor`, or `None` if no loss results (which is
      the case when called by `Model.test_step`).
    """
    del x  # The default implementation does not use `x`.
    return self.loss_container(
        y,
        y_pred,
        sample_weight,
        # regularization_losses=self.losses
    )

  def compute_metrics(self, x, y, y_pred, sample_weight):
    """Update metric states and collect all metrics to be returned.

    Subclasses can optionally override this method to provide custom metric
    updating and collection logic.

    Example:
    ```python
    class MyModel(tf.keras.Sequential):

      def compute_metrics(self, x, y, y_pred, sample_weight):

        # This super call updates `self.compiled_metrics` and returns results
        # for all metrics listed in `self.metrics`.
        metric_results = super(MyModel, self).compute_metrics(
            x, y, y_pred, sample_weight)

        # Note that `self.custom_metric` is not listed in `self.metrics`.
        self.custom_metric.update_state(x, y, y_pred, sample_weight)
        metric_results['custom_metric_name'] = self.custom_metric.result()
        return metric_results
    ```

    Args:
      x: Input data.
      y: Target data.
      y_pred: Predictions returned by the model (output of `model.call(x)`)
      sample_weight: Sample weights for weighting the loss function.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end()`. Typically, the
      values of the metrics listed in `self.metrics` are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    del x  # The default implementation does not use `x`.
    self.metric_container.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return self.get_metrics_result()

  def reset_metrics(self):
    """Resets the state of all the metrics in the model.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> _ = model.fit(x, y, verbose=0)
    >>> assert all(float(m.result()) for m in model.metrics)

    >>> model.reset_metrics()
    >>> assert all(float(m.result()) == 0 for m in model.metrics)

    """
    for m in self.metrics:
      m.reset_state()

  @property
  def metrics_names(self):
    """Returns the model's display labels for all outputs.

    Note: `metrics_names` are available only after a `keras.Model` has been
    trained/evaluated on actual data.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    >>> model.metrics_names
    []

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> model.fit(x, y)
    >>> model.metrics_names
    ['loss', 'mae']

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2, name='out')
    >>> output_1 = d(inputs)
    >>> output_2 = d(inputs)
    >>> model = tf.keras.models.Model(
    ...    inputs=inputs, outputs=[output_1, output_2])
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    >>> model.fit(x, (y, y))
    >>> model.metrics_names
    ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
    'out_1_acc']

    """

    # This property includes all output names including `loss` and per-output
    # losses for backward compatibility.
    return [m.name for m in self.metrics]
