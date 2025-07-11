# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Training-related part of the TF-Keras engine."""

import copy
import os
import random
import sys
import warnings
import weakref
from typing import Union, List, Dict, Text

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import tf_keras as keras
from absl import flags
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tf_keras import callbacks as callbacks_module
from tf_keras import optimizers
from tf_keras.src.dtensor import dtensor_api
from tf_keras.src.dtensor import layout_map as layout_map_lib
from tf_keras.src.engine import compile_utils
from tf_keras.src.engine import data_adapter
from tf_keras.src.engine import training as training_module
from tf_keras.src.engine import training_utils
from tf_keras.src.metrics import base_metric
from tf_keras.src.mixed_precision import loss_scale_optimizer as lso
from tf_keras.src.optimizers import optimizer
from tf_keras.src.optimizers import optimizer_v1
from tf_keras.src.saving import serialization_lib
from tf_keras.src.utils import generic_utils
from tf_keras.src.utils import steps_per_execution_tuning
from tf_keras.src.utils import tf_utils
from tf_keras.src.utils import traceback_utils
from tf_keras.src.utils import version_utils
from tf_keras.src.utils.mode_keys import ModeKeys

from deepray.callbacks import HvdCallbackList
from deepray.callbacks.progbar_logger import ProgbarLogger
from deepray.custom_ops.embedding_variable import kv_variable_ops
from deepray.utils import logging_util
from deepray.utils.horovod_utils import is_main_process

logger = logging_util.get_logger()


def set_random_seed(random_seed):
  random.seed(random_seed)  # set random seed for python
  np.random.seed(random_seed)  # set random seed for numpy
  tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
  os.environ["TF_DETERMINISTIC_OPS"] = "1"  # set random seed for tensorflow-gpu


@keras_export("keras.Model", "keras.models.Model")
class Trainer:
  """A model grouping layers into an object with training/inference features.

  Args:
      inputs: The input(s) of the model: a `keras.Input` object or a
          combination of `keras.Input` objects in a dict, list or tuple.
      outputs: The output(s) of the model: a tensor that originated from
          `keras.Input` objects or a combination of such tensors in a dict,
          list or tuple. See Functional API example below.
      name: String, the name of the model.

  There are two ways to instantiate a `Model`:

  1 - With the "Functional API", where you start from `Input`,
  you chain layer calls to specify the model's forward pass,
  and finally you create your model from inputs and outputs:

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  Note: Only dicts, lists, and tuples of input tensors are supported. Nested
  inputs are not supported (e.g. lists of list or dicts of dict).

  A new Functional API model can also be created by using the
  intermediate tensors. This enables you to quickly extract sub-components
  of the model.

  Example:

  ```python
  inputs = keras.Input(shape=(None, None, 3))
  processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
  conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
  pooling = keras.layers.GlobalAveragePooling2D()(conv)
  feature = keras.layers.Dense(10)(pooling)

  full_model = keras.Model(inputs, feature)
  backbone = keras.Model(processed, conv)
  activations = keras.Model(conv, feature)
  ```

  Note that the `backbone` and `activations` models are not
  created with `keras.Input` objects, but with the tensors that are originated
  from `keras.Input` objects. Under the hood, the layers and weights will
  be shared across these models, so that user can train the `full_model`, and
  use `backbone` or `activations` to do feature extraction.
  The inputs and outputs of the model can be nested structures of tensors as
  well, and the created models are standard Functional API models that support
  all the existing APIs.

  2 - By subclassing the `Model` class: in that case, you should define your
  layers in `__init__()` and you should implement the model's forward pass
  in `call()`.

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)

  model = MyModel()
  ```

  If you subclass `Model`, you can optionally have
  a `training` argument (boolean) in `call()`, which you can use to specify
  a different behavior in training and inference:

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
      x = self.dense1(inputs)
      if training:
        x = self.dropout(x, training=training)
      return self.dense2(x)

  model = MyModel()
  ```

  Once the model is created, you can config the model with losses and metrics
  with `model.compile()`, train the model with `model.fit()`, or use the model
  to do prediction with `model.predict()`.
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  @traceback_utils.filter_traceback
  def __init__(
    self,
    model: Union[keras.Model, List[keras.Model], Dict[Text, keras.Model]],
    optimizer="rmsprop",
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
    pss_evaluation_shards=0,
    *args,
    **kwargs,
  ):
    self._model = {}
    if isinstance(model, list):
      if len(model) > 0:
        self._model = {"main": model[0]}
        if len(model) == 2:
          self._model["sub_model"] = model[1]
        else:
          for i in range(1, len(model)):
            self._model[f"sub_model{i}"] = model[i]
      else:
        raise ValueError("Not a reachable model.")
    elif isinstance(model, dict):
      main_keys = [k for k in model.keys() if "main" in k]
      if len(main_keys) == 1:
        if len(model) == 1:
          self._model = {"main": next(iter(model.values()))}
        else:
          self._model = model
      else:
        raise ValueError(f'Must set only one model with key contains "main", found {main_keys}.')
    elif isinstance(model, (keras.Model, tf.keras.Model)):
      self._model = {"main": model}
    else:
      raise ValueError("Not a reachable model.")

    if run_eagerly is None:
      run_eagerly = flags.FLAGS.run_eagerly

    if steps_per_execution is None:
      steps_per_execution = flags.FLAGS.steps_per_execution

    # Special case for Subclassed Functional Model, which we couldn't detect
    # when __new__ is called. We only realize it is a functional model when
    # it calls super.__init__ with input and output tensor.
    from tf_keras.src.engine import functional

    if training_module.is_functional_model_init_params(args, kwargs) and not isinstance(self, functional.Functional):
      # Filter the kwargs for multiple inheritance.
      supported_kwargs = [
        "inputs",
        "outputs",
        "name",
        "trainable",
        "skip_init",
      ]
      model_kwargs = {k: kwargs[k] for k in kwargs if k in supported_kwargs}
      other_kwargs = {k: kwargs[k] for k in kwargs if k not in supported_kwargs}
      training_module.inject_functional_model_class(self.__class__)
      functional.Functional.__init__(self, *args, **model_kwargs)

      # In case there is any multiple inheritance here, we need to call
      # the __init__ for any class that appears after the Functional
      # class.
      clz_to_init = []
      found_functional_class = False
      for clz in self.__class__.__bases__:
        if issubclass(clz, functional.Functional):
          found_functional_class = True
          continue
        if found_functional_class:
          clz_to_init.append(clz)

      if clz_to_init:
        for clz in clz_to_init:
          clz.__init__(self, *args, **other_kwargs)
      elif other_kwargs:
        # In case there are unused kwargs, we should raise an error to
        # user, in case they have a typo in the param name.
        raise TypeError("The following keyword arguments passed to `Model` aren't supported: {}.".format(other_kwargs))
      return

    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # `inputs` / `outputs` will only appear in kwargs if either are
    # misspelled.
    generic_utils.validate_kwargs(
      kwargs,
      {
        "trainable",
        "dtype",
        "dynamic",
        "name",
        "autocast",
        "inputs",
        "outputs",
      },
    )
    super().__init__(**kwargs)

    # stop_training is used by callback to stop training when error happens
    self.stop_training = False
    self.history = None
    # These objects are used in the default `Model.compile`. They are not
    # guaranteed to be set after `Model.compile` is called, as users can
    # override compile with custom logic.
    self.compiled_loss = None
    self.compiled_metrics = None

    # Don't reset compilation if already done. This may occur if calling
    # `__init__` (or `_init_graph_network`) on an already-compiled model
    # such as a Sequential model. Sequential models may need to rebuild
    # themselves after compilation.
    self._maybe_create_attribute("_is_compiled", False)
    self._maybe_create_attribute("optimizer", None)

    # Model must be created under scope of DistStrat it will be trained
    # with.
    if tf.distribute.has_strategy():
      self._distribution_strategy = tf.distribute.get_strategy()
    else:
      self._distribution_strategy = None
    self._distribute_reduction_method = None

    self._cluster_coordinator = None

    # Defaults to value of `tf.config.experimental_functions_run_eagerly`.
    self._run_eagerly = None
    # Initialize cache attrs.
    self._reset_compile_cache()

    # Fault-tolerance handler. Set in `ModelCheckpoint`.
    self._training_state = None

    self._steps_per_execution = None
    self._steps_per_execution_tuner = None
    self._autotune_steps_per_execution = False

    self._layout_map = layout_map_lib.get_current_layout_map()

    self._init_batch_counters()
    self._base_model_initialized = True

    # `jit_compile` starts off with None as default and gets overwritten by
    # the value specified in `Model.compile`, and this is effective for
    # `fit`, `evaluate`, and `predict`.
    self._jit_compile = None

    self.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      weighted_metrics=weighted_metrics,
      run_eagerly=run_eagerly,
      steps_per_execution=steps_per_execution,
      jit_compile=jit_compile,
      pss_evaluation_shards=pss_evaluation_shards,
      **kwargs,
    )

    if is_main_process():
      logger.info("Initialize training")
      logger.info("flags.FLAGS:")
      for key, value in sorted(flags.FLAGS.flag_values_dict().items()):
        logger.info(f"\t{key:25}= {value}")
    if flags.FLAGS.random_seed is not None:
      set_random_seed(flags.FLAGS.random_seed)

  def _create_counter_variable(self, init_value):
    """Helper function for counter variable creation.

    For the DTensor use case with layout map, since the variable are not
    tracked by model, they can't be visited by the layout map, and need to
    be properly initialized as DVariable.
    """
    # This function should be removed after we move to the strategy based
    # implementation for DTensor.
    if self._layout_map is None:
      agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
      return tf.Variable(init_value, dtype="int64", aggregation=agg)
    else:
      layout = dtensor_api.Layout.replicated(mesh=self._layout_map.get_default_mesh(), rank=0)
      return dtensor_api.DVariable(init_value, dtype="int64", layout=layout)

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _init_batch_counters(self):
    # Untracked Variables, used to keep track of mini-batches seen in `fit`,
    # `evaluate`, and `predict`.
    if not tf.inside_function():
      # Creating variables inside tf.function is not allowed, hence
      # these would otherwise prevent users from creating TF-Keras layers
      # inside tf.function.
      # These variables are not connected to outputs so they have no
      # effect on graph generation anyway.

      self._train_counter = self._create_counter_variable(0)
      self._test_counter = self._create_counter_variable(0)
      self._predict_counter = self._create_counter_variable(0)
      if flags.FLAGS.use_horovod:
        self.first_batch = tf.Variable(True, trainable=False, dtype=tf.bool, name="first_batch")

  @traceback_utils.filter_traceback
  def compile(
    self,
    optimizer="rmsprop",
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
    pss_evaluation_shards=0,
    **kwargs,
  ):
    """Configures the model for training.

    Example:

    ```python
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.FalseNegatives()])
    ```

    Args:
        optimizer: String (name of optimizer) or optimizer instance. See
          `tf.keras.optimizers`.
        loss: Loss function. May be a string (name of loss function), or
          a `tf.keras.losses.Loss` instance. See `tf.keras.losses`. A loss
          function is any callable with the signature `loss = fn(y_true,
          y_pred)`, where `y_true` are the ground truth values, and
          `y_pred` are the model's predictions.
          `y_true` should have shape
          `(batch_size, d0, .. dN)` (except in the case of
          sparse loss functions such as
          sparse categorical crossentropy which expects integer arrays of
          shape `(batch_size, d0, .. dN-1)`).
          `y_pred` should have shape `(batch_size, d0, .. dN)`.
          The loss function should return a float tensor.
          If a custom `Loss` instance is
          used and reduction is set to `None`, return value has shape
          `(batch_size, d0, .. dN-1)` i.e. per-sample or per-timestep loss
          values; otherwise, it is a scalar. If the model has multiple
          outputs, you can use a different loss on each output by passing a
          dictionary or a list of losses. The loss value that will be
          minimized by the model will then be the sum of all individual
          losses, unless `loss_weights` is specified.
        metrics: List of metrics to be evaluated by the model during
          training and testing. Each of this can be a string (name of a
          built-in function), function or a `tf.keras.metrics.Metric`
          instance. See `tf.keras.metrics`. Typically you will use
          `metrics=['accuracy']`.
          A function is any callable with the signature `result = fn(y_true,
          y_pred)`. To specify different metrics for different outputs of a
          multi-output model, you could also pass a dictionary, such as
          `metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}`.
          You can also pass a list to specify a metric or a list of metrics
          for each output, such as
          `metrics=[['accuracy'], ['accuracy', 'mse']]`
          or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
          strings 'accuracy' or 'acc', we convert this to one of
          `tf.keras.metrics.BinaryAccuracy`,
          `tf.keras.metrics.CategoricalAccuracy`,
          `tf.keras.metrics.SparseCategoricalAccuracy` based on the shapes
          of the targets and of the model output. We do a similar
          conversion for the strings 'crossentropy' and 'ce' as well.
          The metrics passed here are evaluated without sample weighting; if
          you would like sample weighting to apply, you can specify your
          metrics via the `weighted_metrics` argument instead.
        loss_weights: Optional list or dictionary specifying scalar
          coefficients (Python floats) to weight the loss contributions of
          different model outputs. The loss value that will be minimized by
          the model will then be the *weighted sum* of all individual
          losses, weighted by the `loss_weights` coefficients.  If a list,
          it is expected to have a 1:1 mapping to the model's outputs. If a
          dict, it is expected to map output names (strings) to scalar
          coefficients.
        weighted_metrics: List of metrics to be evaluated and weighted by
          `sample_weight` or `class_weight` during training and testing.
        run_eagerly: Bool. If `True`, this `Model`'s logic will not be
          wrapped in a `tf.function`. Recommended to leave this as `None`
          unless your `Model` cannot be run inside a `tf.function`.
          `run_eagerly=True` is not supported when using
          `tf.distribute.experimental.ParameterServerStrategy`. Defaults to
           `False`.
        steps_per_execution: Int or `'auto'`. The number of batches to
          run during each `tf.function` call. If set to "auto", keras will
          automatically tune `steps_per_execution` during runtime. Running
          multiple batches inside a single `tf.function` call can greatly
          improve performance on TPUs, when used with distributed strategies
          such as `ParameterServerStrategy`, or with small models with a
          large Python overhead. At most, one full epoch will be run each
          execution. If a number larger than the size of the epoch is
          passed, the execution will be truncated to the size of the epoch.
          Note that if `steps_per_execution` is set to `N`,
          `Callback.on_batch_begin` and `Callback.on_batch_end` methods will
          only be called every `N` batches (i.e. before/after each
          `tf.function` execution). Defaults to `1`.
        jit_compile: If `True`, compile the model training step with XLA.
          [XLA](https://www.tensorflow.org/xla) is an optimizing compiler
          for machine learning.
          `jit_compile` is not enabled for by default.
          Note that `jit_compile=True`
          may not necessarily work for all models.
          For more information on supported operations please refer to the
          [XLA documentation](https://www.tensorflow.org/xla).
          Also refer to
          [known XLA issues](https://www.tensorflow.org/xla/known_issues)
          for more details.
        pss_evaluation_shards: Integer or 'auto'. Used for
          `tf.distribute.ParameterServerStrategy` training only. This arg
          sets the number of shards to split the dataset into, to enable an
          exact visitation guarantee for evaluation, meaning the model will
          be applied to each dataset element exactly once, even if workers
          fail. The dataset must be sharded to ensure separate workers do
          not process the same data. The number of shards should be at least
          the number of workers for good performance. A value of 'auto'
          turns on exact evaluation and uses a heuristic for the number of
          shards based on the number of workers. 0, meaning no
          visitation guarantee is provided. NOTE: Custom implementations of
          `Model.test_step` will be ignored when doing exact evaluation.
          Defaults to `0`.
        **kwargs: Arguments supported for backwards compatibility only.
    """
    if jit_compile and not tf_utils.can_jit_compile(warn=True):
      jit_compile = False
    self._compile_config = serialization_lib.Config(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      weighted_metrics=weighted_metrics,
      run_eagerly=run_eagerly,
      steps_per_execution=steps_per_execution,
      jit_compile=jit_compile,
    )
    with self.distribute_strategy.scope():
      if "experimental_steps_per_execution" in kwargs:
        logging.warning(
          "The argument `steps_per_execution` is no longer "
          "experimental. Pass `steps_per_execution` instead of "
          "`experimental_steps_per_execution`."
        )
        if not steps_per_execution:
          steps_per_execution = kwargs.pop("experimental_steps_per_execution")

      # When compiling from an already-serialized model, we do not want to
      # reapply some processing steps (e.g. metric renaming for
      # multi-output models, which have prefixes added for each
      # corresponding output name).
      from_serialized = kwargs.pop("from_serialized", False)

      self._validate_compile(optimizer, metrics, **kwargs)
      self._run_eagerly = run_eagerly

      self.optimizer = self._get_optimizer(optimizer)
      self.optimizer.global_step = self._train_counter
      self.main_model.optimizer = self.optimizer

      mesh = None
      if self._layout_map is not None:
        mesh = self._layout_map.get_default_mesh()

      if isinstance(loss, compile_utils.LossesContainer):
        self.compiled_loss = loss
      else:
        self.compiled_loss = compile_utils.LossesContainer(
          loss,
          loss_weights,
          output_names=self.main_model.output_names,
          mesh=mesh,
        )
      self.compiled_metrics = compile_utils.MetricsContainer(
        metrics,
        weighted_metrics,
        output_names=self.main_model.output_names,
        from_serialized=from_serialized,
        mesh=mesh,
      )

      if steps_per_execution == "auto":
        if self._steps_per_execution is None:
          self._configure_steps_per_execution(1)
        self._steps_per_execution_tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
          self.optimizer, self._steps_per_execution
        )
        self._autotune_steps_per_execution = True
      else:
        self._configure_steps_per_execution(steps_per_execution or 1)

      self._pss_evaluation_shards = self._infer_exact_eval_shards(pss_evaluation_shards)

      # Initializes attrs that are reset each time `compile` is called.
      self._reset_compile_cache()
      self._is_compiled = True
      self.loss = loss or {}
      if (self._run_eagerly or self.main_model.dynamic) and jit_compile:
        raise ValueError("You cannot enable `run_eagerly` and `jit_compile` at the same time.")
      else:
        self._jit_compile = jit_compile

  def _get_optimizer(self, optimizer):
    """Wraps `optimizer` in `LossScaleOptimizer` if necessary."""

    def _get_single_optimizer(opt):
      opt = optimizers.get(opt)
      if self.main_model.dtype_policy.name == "mixed_float16" and not isinstance(opt, lso.BaseLossScaleOptimizer):
        # Loss scaling is necessary with mixed_float16 for models to
        # converge to the same accuracy as with float32.
        opt = lso.BaseLossScaleOptimizer(opt)
      return opt

    return tf.nest.map_structure(_get_single_optimizer, optimizer)

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _reset_compile_cache(self):
    self.train_function = None
    self.test_function = None
    self.predict_function = None
    # Used to cache the `tf.function`'ed `train_function` to be logged in
    # TensorBoard, since the original `train_function` is not necessarily
    # a `tf.function` (e.g., with ParameterServerStrategy, the
    # `train_function` is a scheduling of the actual training function to a
    # remote worker).
    self.train_tf_function = None

    # Used to cache `trainable` attr of `Layer`s for `fit`.
    self._compiled_trainable_state = self._get_trainable_state()

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _configure_steps_per_execution(self, steps_per_execution):
    self._steps_per_execution = self._create_counter_variable(steps_per_execution)

  @property
  def _should_compute_mask(self):
    return False

  @property
  def metrics(self):
    """Return metrics added using `compile()` or `add_metric()`.

    Note: Metrics passed to `compile()` are available only after a
    `keras.Model` has been trained/evaluated on actual data.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    >>> [m.name for m in model.metrics]
    []

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> model.fit(x, y)
    >>> [m.name for m in model.metrics]
    ['loss', 'mae']

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2, name='out')
    >>> output_1 = d(inputs)
    >>> output_2 = d(inputs)
    >>> model = tf.keras.models.Model(
    ...    inputs=inputs, outputs=[output_1, output_2])
    >>> model.add_metric(
    ...    tf.reduce_sum(output_2), name='mean', aggregation='mean')
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    >>> model.fit(x, (y, y))
    >>> [m.name for m in model.metrics]
    ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
    'out_1_acc', 'mean']

    """
    metrics = []
    if self._is_compiled:
      if self.compiled_loss is not None:
        metrics += self.compiled_loss.metrics
      if self.compiled_metrics is not None:
        metrics += self.compiled_metrics.metrics

    for l in self.main_model._flatten_layers():
      metrics.extend(l._metrics)
    return metrics

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

    # This property includes all output names including `loss` and
    # per-output losses for backward compatibility.
    return [m.name for m in self.metrics]

  @property
  def distribute_strategy(self):
    """The `tf.distribute.Strategy` this model was created under."""
    return self._distribution_strategy or tf.distribute.get_strategy()

  @property
  def run_eagerly(self):
    """Settable attribute indicating whether the model should run eagerly.

    Running eagerly means that your model will be run step by step,
    like Python code. Your model might run slower, but it should become
    easier for you to debug it by stepping into individual layer calls.

    By default, we will attempt to compile your model to a static graph to
    deliver the best execution performance.

    Returns:
      Boolean, whether the model should run eagerly.
    """
    if self.main_model.dynamic and self._run_eagerly == False:
      # TODO(fchollet): consider using py_func to enable this.
      raise ValueError(
        "Your model contains layers that can only be "
        "successfully run in eager execution (layers "
        "constructed with `dynamic=True`). "
        "You cannot set `run_eagerly=False`."
      )

    if self._cluster_coordinator and self._run_eagerly:
      raise ValueError("When using `Model` with `ParameterServerStrategy`, `run_eagerly` is not supported.")

    # Run eagerly logic, by priority:
    # (1) Dynamic models must be run eagerly.
    # (2) Explicitly setting run_eagerly causes a Model to be run eagerly.
    # (3) Not explicitly setting run_eagerly defaults to TF's global
    # setting.
    return (
      self.main_model.dynamic or self._run_eagerly or (tf.config.functions_run_eagerly() and self._run_eagerly is None)
    )

  @run_eagerly.setter
  def run_eagerly(self, value):
    self._run_eagerly = value

  @property
  def autotune_steps_per_execution(self):
    """Settable property to enable tuning for steps_per_execution"""
    return self._autotune_steps_per_execution

  @autotune_steps_per_execution.setter
  def autotune_steps_per_execution(self, value):
    self._autotune_steps_per_execution = value
    if value and self._steps_per_execution_tuner is None:
      if self._steps_per_execution is None:
        self._configure_steps_per_execution(1)
      self._steps_per_execution_tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
        self.optimizer, self._steps_per_execution
      )

  @property
  def steps_per_execution(self):
    """Settable `steps_per_execution variable. Requires a compiled model."""
    return self._steps_per_execution

  @steps_per_execution.setter
  def steps_per_execution(self, value):
    if self._steps_per_execution is None:
      self._configure_steps_per_execution(value)
    else:
      self._steps_per_execution.assign(value)

  @property
  def jit_compile(self):
    """Specify whether to compile the model with XLA.

    [XLA](https://www.tensorflow.org/xla) is an optimizing compiler
    for machine learning. `jit_compile` is not enabled by default.
    Note that `jit_compile=True` may not necessarily work for all models.

    For more information on supported operations please refer to the
    [XLA documentation](https://www.tensorflow.org/xla). Also refer to
    [known XLA issues](https://www.tensorflow.org/xla/known_issues)
    for more details.
    """
    return self._jit_compile

  @jit_compile.setter
  def jit_compile(self, value):
    # Function remains cached with previous jit_compile settings
    if self._jit_compile == value:
      # Avoid resetting compiler cache if possible if the value is the
      # same
      return
    # Check if TensorFlow is compiled with XLA before setting the value
    if value and not tf_utils.can_jit_compile(warn=True):
      self._jit_compile = False
      return

    self._jit_compile = value
    # Setting `jit_compile` should invalidate previously cached functions.
    self._reset_compile_cache()

  @property
  def distribute_reduction_method(self):
    """The method employed to reduce per-replica values during training.

    Unless specified, the value "auto" will be assumed, indicating that
    the reduction strategy should be chosen based on the current
    running environment.
    See `reduce_per_replica` function for more details.

    """
    return self._distribute_reduction_method or "auto"

  @distribute_reduction_method.setter
  def distribute_reduction_method(self, value):
    self._distribute_reduction_method = value

  def _validate_target_and_loss(self, y, loss):
    """Raises error if target or loss is not found.

    This method verifies that the target and loss are properly populated
    when applicable, or raises errors.

    Args:
      y: the target for training.
      loss: the total loss tensor including loss added via `compile` and
        `add_loss`.
    """

    # `self.loss` references the loss added via `compile` call. If users
    # have provided such, the target must be provided; otherwise it's a user
    # error.  Note that `self.loss` does not include losses added via
    # `add_loss`, and it is a valid use when such loss from `add_loss`
    # exists and target does not.
    if self.loss and y is None:
      raise ValueError(
        "Target data is missing. Your model was compiled with "
        f"loss={self.loss}, "
        "and therefore expects target data to be provided in `fit()`."
      )

    # For training, there must be compiled loss or regularization loss to
    # exist in order to apply the gradients. If one is not found, it means
    # no loss was supplied via `compile` or `add_loss`.
    elif loss is None:
      raise ValueError("No loss found. You may have forgotten to provide a `loss` argument in the `compile()` method.")

  def train_step(self, data):
    """The logic for one training step.

    This method can be overridden to support custom training logic.
    For concrete examples of how to override this method see
    [Customizing what happens in fit](
    https://www.tensorflow.org/guide/tf_keras/customizing_what_happens_in_fit).
    This method is called by `Model.make_train_function`.

    This method should contain the mathematical logic for one step of
    training.  This typically includes the forward pass, loss calculation,
    backpropagation, and metric updates.

    Configuration details for *how* this logic is run (e.g. `tf.function`
    and `tf.distribute.Strategy` settings), should be left to
    `Model.make_train_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred = self.main_model(x, training=True)
      loss = self.compute_loss(x, y, y_pred, sample_weight)
    self._validate_target_and_loss(y, loss)
    # Run backwards pass.
    self.optimizer.minimize(loss, self.main_model.trainable_variables, tape=tape)
    return self.compute_metrics(x, y, y_pred, sample_weight)

  def hvd_train_step(self, data):
    """The logic for one training step on Horovod.

    This method can be overridden to support custom training logic.
    For concrete examples of how to override this method see
    [Customizing what happens in fit](
    https://www.tensorflow.org/guide/tf_keras/customizing_what_happens_in_fit).
    This method is called by `Model.make_train_function`.

    This method should contain the mathematical logic for one step of
    training.  This typically includes the forward pass, loss calculation,
    backpropagation, and metric updates.

    Configuration details for *how* this logic is run (e.g. `tf.function`
    and `tf.distribute.Strategy` settings), should be left to
    `Model.make_train_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    # Run forward pass.
    with tf.GradientTape() as dp_tape, tf.GradientTape() as mp_tape:
      y_pred = self.main_model(x, training=True)
      loss = self.compute_loss(x, y, y_pred, sample_weight)
    dp_tape = hvd.DistributedGradientTape(dp_tape, sparse_as_dense=False)
    self._validate_target_and_loss(y, loss)
    # Run backwards pass.
    dp_vars, mp_vars = [], []
    for x in self.main_model.variables:
      if isinstance(x, kv_variable_ops.EmbeddingVariable):
        mp_vars.append(x)
      else:
        dp_vars.append(x)
    self.optimizer.minimize(loss, dp_vars, tape=dp_tape)
    self.optimizer.minimize(loss, mp_vars, tape=mp_tape)
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
      The total loss as a `tf.Tensor`, or `None` if no loss results (which
      is the case when called by `Model.test_step`).
    """
    del x  # The default implementation does not use `x`.
    return self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.main_model.losses)

  def compute_metrics(self, x, y, y_pred, sample_weight):
    """Update metric states and collect all metrics to be returned.

    Subclasses can optionally override this method to provide custom metric
    updating and collection logic.

    Example:
    ```python
    class MyModel(tf.keras.Sequential):

      def compute_metrics(self, x, y, y_pred, sample_weight):

        # This super call updates `self.compiled_metrics` and returns
        # results for all metrics listed in `self.metrics`.
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
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return self.get_metrics_result()

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

  def _validate_and_get_metrics_result(self, logs):
    """Returns model metrics as a dict if the keys match with input logs.

    When the training / evalution is performed with asynchronous steps, such
    as the case with `tf.distribute.ParameterServerStrategy`, the last
    scheduled `train / test_step` may not give the latest metrics because it
    is not guaranteed to be executed the last. This method gets metrics from
    the model directly instead of relying on the return from last step
    function.

    It logs a warning if the metric results could not be overridden when
    used with `tf.distribute.ParameterServerStrategy`.

    When the user has custom train / test step functions, the metrics
    returned may be different from `Model.metrics`. In those instances,
    this function will be no-op and return the logs.

    Args:
      logs: A `dict` of metrics returned by train / test step function.

    Returns:
      A `dict` containing values of the metrics listed in `self.metrics`
      when logs and model metrics keys match. Otherwise it returns input
      `logs`.
    """
    PSS_WARN_MSG = "Could not get Model metric results. \
        Using the results of last step function could lead to incorrect \
        results when used with ParameterServerStrategy"

    try:
      metric_logs = self.get_metrics_result()
    except TypeError:
      if self._cluster_coordinator:
        logging.warning(PSS_WARN_MSG)
    else:
      # Verify that train / test step logs passed and metric logs have
      # matching keys. Could be different when using custom step functions
      if isinstance(logs, dict) and set(logs.keys()) == set(metric_logs.keys()):
        logs = tf_utils.sync_to_numpy_or_python_type(metric_logs)
      elif self._cluster_coordinator:
        logging.warning(PSS_WARN_MSG)
    return logs

  def _aggregate_exact_metrics(self, logs):
    # When doing exact evaluation, `logs` is a list of each data shard's
    # metric variables, which will be used to update the metrics.
    for shard_result in logs:
      for metric in self.metrics:
        if metric.name not in shard_result.keys():
          logging.log_first_n(
            logging.WARN,
            f"No matching result found for metric {metric.name}. This metric's computed result may be incorrect.",
            3,
          )
          continue
        metric_result = shard_result[metric.name]
        if len(metric_result) != len(metric.weights):
          raise ValueError(
            f"Expected {len(metric.weights)} variables in result "
            f"for metric {metric.name}, but found "
            f"{len(metric_result)}."
          )
        for weight, val in zip(metric.weights, metric_result):
          weight.assign_add(val)
    return self.get_metrics_result()

  def make_train_function(self, force=False):
    """Creates a function that executes one step of training.

    This method can be overridden to support custom training logic.
    This method is called by `Model.fit` and `Model.train_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual training
    logic to `Model.train_step`.

    This function is cached the first time `Model.fit` or
    `Model.train_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called. You can skip the cache and generate again the
    function with `force=True`.

    Args:
      force: Whether to regenerate the train function and skip the cached
        function if available.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self.train_function is not None and not force:
      return self.train_function

    def step_function(iterator):
      """Runs a single training step."""

      def run_step(data):
        outputs = self.train_step(data)
        # Ensure counter is updated only if `train_step` succeeds.
        with tf.control_dependencies(training_module._minimum_control_deps(outputs)):
          self._train_counter.assign_add(1)
        return outputs

      if self.jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)
      data = next(iterator)
      outputs = self.distribute_strategy.run(run_step, args=(data,))
      outputs = training_module.reduce_per_replica(
        outputs,
        self.distribute_strategy,
        reduction=self.distribute_reduction_method,
      )
      return outputs

    # Special case if steps_per_execution is one.
    if (
      self._steps_per_execution is None
      or self._steps_per_execution.numpy().item() == 1
      and not self.autotune_steps_per_execution
    ):

      def train_function(iterator):
        """Runs a training execution with a single step."""
        return step_function(iterator)

      if not self.run_eagerly:
        train_function = tf.function(train_function, reduce_retracing=True)
        self.train_tf_function = train_function

      if self._cluster_coordinator:
        self.train_function = lambda it: self._cluster_coordinator.schedule(train_function, args=(it,))
      else:
        self.train_function = train_function

    # If we're using a coordinator, use the value of
    # self._steps_per_execution at the time the function is
    # called/scheduled, and not when it is actually executed.
    elif self._cluster_coordinator:

      def train_function(iterator, steps_per_execution):
        """Runs a training execution with multiple steps."""
        for _ in tf.range(steps_per_execution):
          outputs = step_function(iterator)
        return outputs

      if not self.run_eagerly:
        train_function = tf.function(train_function, reduce_retracing=True)
        self.train_tf_function = train_function

      self.train_function = lambda it: self._cluster_coordinator.schedule(
        train_function, args=(it, self._steps_per_execution.value())
      )
    else:

      def train_function(iterator):
        """Runs a training execution with multiple steps."""
        for _ in tf.range(self._steps_per_execution):
          outputs = step_function(iterator)
        return outputs

      if not self.run_eagerly:
        train_function = tf.function(train_function, reduce_retracing=True)
        self.train_tf_function = train_function
      self.train_function = train_function

    return self.train_function

  def make_hvd_train_function(self, force=False):
    """Creates a function that executes one step of training.

    This method can be overridden to support custom training logic.
    This method is called by `Model.fit` and `Model.train_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual training
    logic to `Model.train_step`.

    This function is cached the first time `Model.fit` or
    `Model.train_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called. You can skip the cache and generate again the
    function with `force=True`.

    Args:
      force: Whether to regenerate the train function and skip the cached
        function if available.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self.train_function is not None and not force:
      return self.train_function

    def step_function(iterator):
      """Runs a single training step."""

      def do_broadcast():
        if flags.FLAGS.use_dynamic_embedding:
          from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
            TrainableWrapper,
            DEResourceVariable,
          )
        else:
          TrainableWrapper, DEResourceVariable = None, None

        # Define the types to check against, including potentially None values
        types_to_exclude = [TrainableWrapper, DEResourceVariable, kv_variable_ops.EmbeddingVariable]
        # Filter out any entries that are None to create a valid tuple of types
        valid_types_to_exclude = tuple(t for t in types_to_exclude if t is not None)

        model_broadcast_vars = [x for x in self.main_model.variables if not isinstance(x, valid_types_to_exclude)]
        opt_broadcast_vars = [x for x in self.optimizer.variables() if not isinstance(x, valid_types_to_exclude)]
        tf.print(
          f"Broadcasting {len(model_broadcast_vars)} model variables & {len(opt_broadcast_vars)} optimizer variables...",
          output_stream=sys.stdout,
        )
        broadcast_op = hvd.broadcast_variables(model_broadcast_vars + opt_broadcast_vars, root_rank=0)
        with tf.control_dependencies([broadcast_op]):
          self.first_batch.assign(False)

      def run_step(data):
        outputs = self.hvd_train_step(data)
        # Ensure counter is updated only if `hvd_train_step` succeeds.
        with tf.control_dependencies(training_module._minimum_control_deps(outputs)):
          self._train_counter.assign_add(1)
          if self.first_batch:
            do_broadcast()
        return outputs

      if self.jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)
      data = next(iterator)
      outputs = run_step(data)
      return outputs

    # Special case if steps_per_execution is one.
    if (
      self._steps_per_execution is None
      or self._steps_per_execution.numpy().item() == 1
      and not self.autotune_steps_per_execution
    ):

      def train_function(iterator):
        """Runs a training execution with a single step."""
        return step_function(iterator)

      if not self.run_eagerly:
        train_function = tf.function(train_function, reduce_retracing=True)
        self.train_tf_function = train_function

      self.train_function = train_function
    else:

      def train_function(iterator):
        """Runs a training execution with multiple steps."""
        for _ in tf.range(self._steps_per_execution):
          outputs = step_function(iterator)
        return outputs

      if not self.run_eagerly:
        train_function = tf.function(train_function, reduce_retracing=True)
        self.train_tf_function = train_function
      self.train_function = train_function

    return self.train_function

  @traceback_utils.filter_traceback
  def fit(
    self,
    x=None,
    y=None,
    batch_size=None,
    epochs=None,
    verbose="auto",
    callbacks=[],
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
  ):
    """Trains the model for a fixed number of epochs (dataset iterations).

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs,
            targets)` or `(inputs, targets, sample_weights)`.
          - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
            callable that takes a single argument of type
            `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
            `DatasetCreator` should be used when users prefer to specify the
            per-replica batching and sharding logic for the `Dataset`.
            See `tf.keras.utils.experimental.DatasetCreator` doc for more
            information.
          A more detailed description of unpacking behavior for iterator
          types (Dataset, generator, Sequence) is given below. If these
          include `sample_weights` as a third component, note that sample
          weighting applies to the `weighted_metrics` argument but not the
          `metrics` argument in `compile()`. If using
          `tf.distribute.experimental.ParameterServerStrategy`, only
          `DatasetCreator` type is supported for `x`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence`
            instances (since they generate batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided
            (unless the `steps_per_epoch` flag is set to
            something other than None).
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 'auto', 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' becomes 1 for most cases, but 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so verbose=2 is
            recommended when not running interactively (eg, in a production
            environment). Defaults to 'auto'.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`. Note
            `tf.keras.callbacks.ProgbarLogger` and
            `tf.keras.callbacks.History` callbacks are created automatically
            and need not be passed into `model.fit`.
            `tf.keras.callbacks.ProgbarLogger` is created or not based on
            `verbose` argument to `model.fit`.
            Callbacks with batch-level calls are currently unsupported with
            `tf.distribute.experimental.ParameterServerStrategy`, and users
            are advised to implement epoch-level calls instead with an
            appropriate `steps_per_epoch` value.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This
            argument is not supported when `x` is a dataset, generator or
            `keras.utils.Sequence` instance.
            If both `validation_data` and `validation_split` are provided,
            `validation_data` will override `validation_split`.
            `validation_split` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact
            that the validation loss of data provided using
            `validation_split` or `validation_data` is not affected by
            regularization layers like noise and dropout.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
              - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                arrays.
              - A `tf.data.Dataset`.
              - A Python generator or `keras.utils.Sequence` returning
              `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            `validation_data` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is
            ignored when `x` is a generator or an object of tf.data.Dataset.
            'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class. When `class_weight` is specified
            and targets have a rank of 2 or greater, either `y` must be
            one-hot encoded, or an explicit final dimension of `1` must
            be included for sparse class labels.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            This argument is not supported when `x` is a dataset, generator,
            or `keras.utils.Sequence` instance, instead provide the
            sample_weights as the third element of `x`.
            Note that sample weighting does not apply to metrics specified
            via the `metrics` argument in `compile()`. To apply sample
            weighting to your metrics, you can specify them via the
            `weighted_metrics` in `compile()` instead.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is
            exhausted.  When passing an infinitely repeating dataset, you
            must specify the `steps_per_epoch` argument. If
            `steps_per_epoch=-1` the training will run indefinitely with an
            infinitely repeating dataset.  This argument is not supported
            with array inputs.
            When using `tf.distribute.experimental.ParameterServerStrategy`:
              * `steps_per_epoch=None` is not supported.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None,
            validation will run until the `validation_data` dataset is
            exhausted. In the case of an infinitely repeated dataset, it
            will run into an infinite loop. If 'validation_steps' is
            specified and only part of the dataset will be consumed, the
            evaluation will start from the beginning of the dataset at each
            epoch. This ensures that the same validation samples are used
            every time.
        validation_batch_size: Integer or `None`.
            Number of samples per validation batch.
            If unspecified, will default to `batch_size`.
            Do not specify the `validation_batch_size` if your data is in
            the form of datasets, generators, or `keras.utils.Sequence`
            instances (since they generate batches).
        validation_freq: Only relevant if validation data is provided.
          Integer or `collections.abc.Container` instance (e.g. list, tuple,
          etc.).  If an integer, specifies how many training epochs to run
          before a new validation run is performed, e.g. `validation_freq=2`
          runs validation every 2 epochs. If a Container, specifies the
          epochs on which to run validation, e.g.
          `validation_freq=[1, 2, 10]` runs validation at the end of the
          1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or
          `keras.utils.Sequence` input only. Maximum size for the generator
          queue.  If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-pickleable arguments to
            the generator as they can't be passed easily to children
            processes.

    Unpacking behavior for iterator-like inputs:
        A common pattern is to pass a tf.data.Dataset, generator, or
      tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
      yield not only features (x) but optionally targets (y) and sample
      weights.  TF-Keras requires that the output of such iterator-likes be
      unambiguous. The iterator should return a tuple of length 1, 2, or 3,
      where the optional second and third elements will be used for y and
      sample_weight respectively. Any other type provided will be wrapped in
      a length one tuple, effectively treating everything as 'x'. When
      yielding dicts, they should still adhere to the top-level tuple
      structure.
      e.g. `({"x0": x0, "x1": x1}, y)`. TF-Keras will not attempt to
      separate features, targets, and weights from the keys of a single
      dict.
        A notable unsupported data type is the namedtuple. The reason is
      that it behaves like both an ordered datatype (tuple) and a mapping
      datatype (dict). So given a namedtuple of the form:
          `namedtuple("example_tuple", ["y", "x"])`
      it is ambiguous whether to reverse the order of the elements when
      interpreting the value. Even worse is a tuple of the form:
          `namedtuple("other_tuple", ["x", "y", "z"])`
      where it is unclear if the tuple was intended to be unpacked into x,
      y, and sample_weight or passed through as a single element to `x`. As
      a result the data processing code will simply raise a ValueError if it
      encounters a namedtuple. (Along with instructions to remedy the
      issue.)

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: 1. If the model was never compiled or,
        2. If `model.fit` is  wrapped in `tf.function`.

        ValueError: In case of mismatch between the provided input data
            and what the model expects or when the input data is empty.
    """
    if steps_per_epoch and flags.FLAGS.use_horovod:
      try:
        import horovod.tensorflow as hvd

        logger.debug(f"steps_per_epoch = {steps_per_epoch}")
        steps_array = hvd.allgather_object(steps_per_epoch, name="check_train_step")
        logger.debug(f"steps_array = {steps_array}")
        assert max(set(steps_array)) == min(set(steps_array))
      except Exception as e:
        logger.exception(e)
        raise ValueError(
          f"steps_per_epoch = {steps_per_epoch}, different rank should have same steps when using Horovod."
        )
    # Legacy graph support is contained in `training_v1.Model`.
    if batch_size is None:
      batch_size = flags.FLAGS.batch_size
    if epochs is None:
      epochs = flags.FLAGS.epochs
    if flags.FLAGS.stop_steps >= 0:
      epochs = 1
      if steps_per_epoch is None:
        steps_per_epoch = flags.FLAGS.stop_steps
      else:
        steps_per_epoch = min(steps_per_epoch, flags.FLAGS.stop_steps)

    version_utils.disallow_legacy_graph("Model", "fit")
    self._assert_compile_was_called()
    self._check_call_args("fit")
    training_module._disallow_inside_tf_function("fit")

    verbose = training_module._get_verbosity(verbose, self.distribute_strategy)

    if validation_split and validation_data is None:
      # Create the validation data using the training data. Only supported
      # for `Tensor` and `NumPy` input.
      (
        (
          x,
          y,
          sample_weight,
        ),
        validation_data,
      ) = data_adapter.train_validation_split((x, y, sample_weight), validation_split=validation_split)

    if validation_data:
      (
        val_x,
        val_y,
        val_sample_weight,
      ) = data_adapter.unpack_x_y_sample_weight(validation_data)

    if self.distribute_strategy._should_use_with_coordinator:
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(self.distribute_strategy)

    with (
      self.distribute_strategy.scope(),
      training_utils.RespectCompiledTrainableState(  # noqa: E501
        self
      ),
    ):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.get_data_handler(
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=epochs,
        shuffle=shuffle,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        model=self,
        steps_per_execution=self._steps_per_execution,
      )

      for callback in callbacks:
        if hasattr(callback, "set_optimizer") and callable(callback.set_optimizer):
          callback.set_optimizer(self.optimizer)
        if hasattr(callback, "set_models") and callable(callback.set_models):
          callback.set_models(self._model)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        if flags.FLAGS.use_horovod:
          if is_main_process():
            callbacks += [ProgbarLogger(count_mode="steps")]
          callbacks = HvdCallbackList(
            callbacks,
            add_history=True,
            add_progbar=False,
            model=self.main_model,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps,
          )
        else:
          callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self.main_model,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps,
          )

      self.stop_training = False
      self.train_function = (
        self.make_train_function() if not flags.FLAGS.use_horovod else self.make_hvd_train_function()
      )
      self._train_counter.assign(0)
      callbacks.on_train_begin()
      training_logs = None
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.start()
      # Handle fault-tolerance for multi-worker.
      # TODO(omalleyt): Fix the ordering issues that mean this has to
      # happen after `callbacks.on_train_begin`.
      steps_per_epoch_inferred = steps_per_epoch or data_handler.inferred_steps
      (
        data_handler._initial_epoch,
        data_handler._initial_step,
      ) = self._maybe_load_initial_counters_from_ckpt(steps_per_epoch_inferred, initial_epoch)
      logs = None
      for epoch, iterator in data_handler.enumerate_epochs():
        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with tf.profiler.experimental.Trace(
              "train",
              epoch_num=epoch,
              step_num=step,
              batch_size=batch_size,
              _r=1,
            ):
              callbacks.on_train_batch_begin(step)
              tmp_logs = self.train_function(iterator)
              if data_handler.should_sync:
                context.async_wait()
              # No error, now safe to assign to logs.
              logs = tmp_logs
              end_step = step + data_handler.step_increment
              callbacks.on_train_batch_end(end_step, logs)
              if self.stop_training:
                break

        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        if logs is None:
          raise ValueError(
            "Unexpected result of `train_function` "
            "(Empty logs). This could be due to issues in input "
            "pipeline that resulted in an empty dataset. "
            "Otherwise, please use "
            "`Model.compile(..., run_eagerly=True)`, or "
            "`tf.config.run_functions_eagerly(True)` for more "
            "information of where went wrong, or file a "
            "issue/bug to `tf.keras`."
          )
        # Override with model metrics instead of last step logs
        logs = self._validate_and_get_metrics_result(logs)
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          if self._pss_evaluation_shards:
            self._disallow_exact_eval_with_add_metrics()
          # Create data_handler for evaluation and cache it.
          if getattr(self, "_eval_data_handler", None) is None:
            self._eval_data_handler = data_adapter.get_data_handler(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size,
              steps_per_epoch=validation_steps,
              initial_epoch=0,
              epochs=1,
              max_queue_size=max_queue_size,
              workers=workers,
              use_multiprocessing=use_multiprocessing,
              model=self,
              steps_per_execution=self._steps_per_execution,
              pss_evaluation_shards=self._pss_evaluation_shards,
            )
          val_logs = self.evaluate(
            x=val_x,
            y=val_y,
            sample_weight=val_sample_weight,
            batch_size=validation_batch_size or batch_size,
            steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            return_dict=True,
            _use_cached_eval_dataset=True,
          )
          val_logs = {"val_" + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if self.stop_training:
          break

      if isinstance(self.optimizer, optimizer.Optimizer) and epochs > 0:
        self.optimizer.finalize_variable_values(self.main_model.trainable_variables)

      # If eval data_handler exists, delete it after all epochs are done.
      if getattr(self, "_eval_data_handler", None) is not None:
        del self._eval_data_handler
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.stop()
      callbacks.on_train_end(logs=training_logs)
      return self.history

  def test_step(self, data):
    """The logic for one evaluation step.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.make_test_function`.

    This function should contain the mathematical logic for one step of
    evaluation.
    This typically includes the forward pass, loss calculation, and metrics
    updates.

    Configuration details for *how* this logic is run (e.g. `tf.function`
    and `tf.distribute.Strategy` settings), should be left to
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

  def _make_test_function_exact(self):
    if getattr(self, "_shard_test_function", None):
      return self._shard_test_function

    def step_function(batch):
      def run_step(data):
        # TODO(b/272050910): Use sample_weight for weighted metrics.
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self.main_model(x, training=False)
        return x, y, y_pred, sample_weight

      if self._jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)

      outputs = self.distribute_strategy.run(run_step, args=(batch,))
      outputs = training_module.reduce_per_replica(
        outputs,
        self.distribute_strategy,
        reduction=self.distribute_reduction_method,
      )
      return outputs

    def shard_test_function(dataset, total_shards, shard_idx):
      # Copy loss and metric variables to the worker and work with them
      # locally. This ensures each shard function is atomic: if a worker
      # is preempted, the intermediate progress is discarded and that
      # shard is retried. This in turn guarantees exactly-once visitation.
      local_unweighted_metrics, local_weighted_metrics = [], []
      with tf_utils.with_metric_local_vars_scope():
        # TODO(jmullenbach): implement and use a clone for
        # `MetricsContainer` and use its `update_state` method directly.
        for metric in self.compiled_metrics.unweighted_metrics:
          if metric is not None:
            local_unweighted_metrics.append(base_metric.clone_metric(metric))
        for metric in self.compiled_metrics.weighted_metrics:
          if metric is not None:
            local_weighted_metrics.append(base_metric.clone_metric(metric))
        local_loss = compile_utils.LossesContainer.from_config(self.compiled_loss.get_config())

      dataset = input_ops.auto_shard_dataset(dataset, total_shards, shard_idx)
      iterator = iter(dataset)
      with distribute_utils.cache_variable_reads():
        for batch in iterator:
          x, y, y_pred, sample_weight = step_function(batch)
          for weighted_metric in local_weighted_metrics:
            weighted_metric.update_state(y, y_pred, sample_weight)
          for unweighted_metric in local_unweighted_metrics:
            unweighted_metric.update_state(y, y_pred)
          local_loss(y, y_pred, sample_weight)
      local_metrics = local_unweighted_metrics + local_weighted_metrics + local_loss.metrics
      outputs = {metric.name: metric.weights for metric in local_metrics}
      with tf.control_dependencies(training_module._minimum_control_deps(outputs)):
        self._test_counter.assign_add(1)
      return outputs

    if not self.run_eagerly:
      shard_test_function = tf.function(shard_test_function, reduce_retracing=True)

    self._shard_test_function = lambda *args: self._cluster_coordinator.schedule(
      shard_test_function,
      args=args,
    )
    return self._shard_test_function

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

    def step_function(iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = self.test_step(data)
        # Ensure counter is updated only if `test_step` succeeds.
        with tf.control_dependencies(training_module._minimum_control_deps(outputs)):
          self._test_counter.assign_add(1)
        return outputs

      if self.jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)

      data = next(iterator)
      outputs = self.distribute_strategy.run(run_step, args=(data,))
      outputs = training_module.reduce_per_replica(
        outputs,
        self.distribute_strategy,
        reduction=self.distribute_reduction_method,
      )
      return outputs

    # Special case if steps_per_execution is one.
    if (
      self._steps_per_execution is None
      or self._steps_per_execution.numpy().item() == 1
      and not self.autotune_steps_per_execution
    ):

      def test_function(iterator):
        """Runs a test execution with a single step."""
        return step_function(iterator)

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)

      if self._cluster_coordinator:
        self.test_function = lambda it: self._cluster_coordinator.schedule(test_function, args=(it,))
      else:
        self.test_function = test_function

    # If we're using a coordinator, use the value of
    # self._steps_per_execution at the time the function is
    # called/scheduled, and not when it is actually executed.
    elif self._cluster_coordinator:

      def test_function(iterator, steps_per_execution):
        """Runs a test execution with multiple steps."""
        for _ in tf.range(steps_per_execution):
          outputs = step_function(iterator)
        return outputs

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)

      self.test_function = lambda it: self._cluster_coordinator.schedule(
        test_function, args=(it, self._steps_per_execution.value())
      )
    else:

      def test_function(iterator):
        """Runs a test execution with multiple steps."""
        for _ in tf.range(self._steps_per_execution):
          outputs = step_function(iterator)
        return outputs

      if not self.run_eagerly:
        test_function = tf.function(test_function, reduce_retracing=True)
      self.test_function = test_function

    return self.test_function

  @traceback_utils.filter_traceback
  def evaluate(
    self,
    x=None,
    y=None,
    batch_size=None,
    verbose="auto",
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False,
    **kwargs,
  ):
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs,
            targets)` or `(inputs, targets, sample_weights)`.
          A more detailed description of unpacking behavior for iterator
          types (Dataset, generator, Sequence) is given in the `Unpacking
          behavior for iterator-like inputs` section of `Model.fit`.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely).
          If `x` is a dataset, generator or `keras.utils.Sequence` instance,
          `y` should not be specified (since targets will be obtained from
          the iterator/dataset).
        batch_size: Integer or `None`. Number of samples per batch of
          computation. If unspecified, `batch_size` will default to 32. Do
          not specify the `batch_size` if your data is in the form of a
          dataset, generators, or `keras.utils.Sequence` instances (since
          they generate batches).
        verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = single line.
            `"auto"` becomes 1 for most cases, and to 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so `verbose=2` is
            recommended when not running interactively (e.g. in a production
            environment). Defaults to 'auto'.
        sample_weight: Optional Numpy array of weights for the test samples,
          used for weighting the loss function. You can either pass a flat
          (1D) Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples), or in the case of
              temporal data, you can pass a 2D array with shape `(samples,
              sequence_length)`, to apply a different weight to every
              timestep of every sample. This argument is not supported when
              `x` is a dataset, instead pass sample weights as the third
              element of `x`.
        steps: Integer or `None`. Total number of steps (batches of samples)
          before declaring the evaluation round finished. Ignored with the
          default value of `None`. If x is a `tf.data` dataset and `steps`
          is None, 'evaluate' will run until the dataset is exhausted. This
          argument is not supported with array inputs.
        callbacks: List of `keras.callbacks.Callback` instances. List of
          callbacks to apply during evaluation. See
          [callbacks](https://www.tensorflow.org/api_docs/python/tf/tf_keras/callbacks).
        max_queue_size: Integer. Used for generator or
          `keras.utils.Sequence` input only. Maximum size for the generator
          queue. If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
          only. Maximum number of processes to spin up when using
          process-based threading. If unspecified, `workers` will default to
          1.
        use_multiprocessing: Boolean. Used for generator or
          `keras.utils.Sequence` input only. If `True`, use process-based
          threading. If unspecified, `use_multiprocessing` will default to
          `False`. Note that because this implementation relies on
          multiprocessing, you should not pass non-pickleable arguments to
          the generator as they can't be passed easily to children
          processes.
        return_dict: If `True`, loss and metric results are returned as a
          dict, with each key being the name of the metric. If `False`, they
          are returned as a list.
        **kwargs: Unused at this time.

    See the discussion of `Unpacking behavior for iterator-like inputs` for
    `Model.fit`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: If `model.evaluate` is wrapped in a `tf.function`.
    """
    version_utils.disallow_legacy_graph("Model", "evaluate")
    self._assert_compile_was_called()
    self._check_call_args("evaluate")
    self._check_sample_weight_warning(x, sample_weight)
    training_module._disallow_inside_tf_function("evaluate")
    use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
    if kwargs:
      raise TypeError(f"Invalid keyword arguments: {list(kwargs.keys())}")

    if self.distribute_strategy._should_use_with_coordinator:
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(self.distribute_strategy)

    verbose = training_module._get_verbosity(verbose, self.distribute_strategy)
    if self._pss_evaluation_shards:
      self._disallow_exact_eval_with_add_metrics()
    with self.distribute_strategy.scope():
      # Use cached evaluation data only when it's called in `Model.fit`
      if use_cached_eval_dataset and getattr(self, "_eval_data_handler", None) is not None:
        data_handler = self._eval_data_handler
      else:
        # Creates a `tf.data.Dataset` and handles batch and epoch
        # iteration.
        data_handler = data_adapter.get_data_handler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution,
          pss_evaluation_shards=self._pss_evaluation_shards,
        )

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
          callbacks,
          add_history=True,
          add_progbar=verbose != 0,
          model=self,
          verbose=verbose,
          epochs=1,
          steps=data_handler.inferred_steps,
        )

      # Initialize to prevent errors if 0 epochs are evaluated.
      logs = {}

      test_function_runner = self._get_test_function_runner(callbacks)
      self._test_counter.assign(0)
      callbacks.on_test_begin()
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.start()
      for (
        _,
        dataset_or_iterator,
      ) in data_handler.enumerate_epochs():  # Single epoch.
        self.reset_metrics()
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with tf.profiler.experimental.Trace("test", step_num=step, _r=1):
              callbacks.on_test_batch_begin(step)
              logs = test_function_runner.run_step(
                dataset_or_iterator,
                data_handler,
                step,
                self._pss_evaluation_shards,
              )

      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      # Override with model metrics instead of last step logs
      if self._pss_evaluation_shards:
        logs = self._aggregate_exact_metrics(logs)
      else:
        logs = self._validate_and_get_metrics_result(logs)
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.stop()
      callbacks.on_test_end(logs=logs)

      if return_dict:
        return logs
      else:
        return training_module.flatten_metrics_in_order(logs, self.metrics_names)

  def _disallow_exact_eval_with_add_metrics(self):
    metrics_from_add_metric = [metric for layer in self._flatten_layers() for metric in layer._metrics]
    compiled_metrics = self.compiled_metrics.metrics
    if any([metric not in compiled_metrics for metric in metrics_from_add_metric]):
      raise ValueError(
        "Detected that a metric was added to this model "
        "via `Model.add_metric`. This is not currently "
        "supported when using exact evaluation with "
        "`tf.distribute.ParameterServerStrategy`."
      )

  def _infer_exact_eval_shards(self, pss_evaluation_shards):
    if not self.distribute_strategy._should_use_with_coordinator:
      return 0
    if pss_evaluation_shards == "auto":
      # TODO(b/264265138) evaluate and improve this heuristic
      return self.distribute_strategy._num_workers * 5
    return pss_evaluation_shards

  def _get_test_function_runner(self, callbacks):
    if self._pss_evaluation_shards and self.distribute_strategy._should_use_with_coordinator:
      self.test_function = self._make_test_function_exact()
      test_function_runner = training_module._ExactTestFunction(self.test_function, callbacks)
    else:
      self.test_function = self.make_test_function()
      test_function_runner = training_module._TestFunction(self.test_function, callbacks)
    return test_function_runner

  def predict_step(self, data):
    """The logic for one inference step.

    This method can be overridden to support custom inference logic.
    This method is called by `Model.make_predict_function`.

    This method should contain the mathematical logic for one step of
    inference.  This typically includes the forward pass.

    Configuration details for *how* this logic is run (e.g. `tf.function`
    and `tf.distribute.Strategy` settings), should be left to
    `Model.make_predict_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      The result of one inference step, typically the output of calling the
      `Model` on data.
    """
    x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
    return self.main_model(x, training=False)

  def make_predict_function(self, force=False):
    """Creates a function that executes one step of inference.

    This method can be overridden to support custom inference logic.
    This method is called by `Model.predict` and `Model.predict_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.predict_step`.

    This function is cached the first time `Model.predict` or
    `Model.predict_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called. You can skip the cache and generate again the
    function with `force=True`.

    Args:
      force: Whether to regenerate the predict function and skip the cached
        function if available.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return the outputs of the `Model`.
    """
    if self.predict_function is not None and not force:
      return self.predict_function

    def step_function(iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = self.predict_step(data)
        # Ensure counter is updated only if `test_step` succeeds.
        with tf.control_dependencies(training_module._minimum_control_deps(outputs)):
          self._predict_counter.assign_add(1)
        return outputs

      if self.jit_compile:
        run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)

      data = next(iterator)
      outputs = self.distribute_strategy.run(run_step, args=(data,))
      outputs = training_module.reduce_per_replica(outputs, self.distribute_strategy, reduction="concat")
      return outputs

    # Special case if steps_per_execution is one.
    if (
      self._steps_per_execution is None
      or self._steps_per_execution.numpy().item() == 1
      and not self.autotune_steps_per_execution
    ):

      def predict_function(iterator):
        """Runs an evaluation execution with a single step."""
        return step_function(iterator)

    else:

      def predict_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        outputs = step_function(iterator)
        for _ in tf.range(self._steps_per_execution - 1):
          tf.autograph.experimental.set_loop_options(
            shape_invariants=[
              (
                outputs,
                tf.nest.map_structure(
                  lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=True).shape,
                  outputs,
                ),
              )
            ]
          )
          step_outputs = step_function(iterator)
          outputs = tf.nest.map_structure(lambda t1, t2: training_module.concat([t1, t2]), outputs, step_outputs)
        return outputs

    if not self.run_eagerly:
      predict_function = tf.function(predict_function, reduce_retracing=True)
    self.predict_function = predict_function

    return self.predict_function

  @traceback_utils.filter_traceback
  def predict(
    self,
    x,
    batch_size=None,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
  ):
    """Generates output predictions for the input samples.

    Computation is done in batches. This method is designed for batch
    processing of large numbers of inputs. It is not intended for use inside
    of loops that iterate over your data and process small numbers of inputs
    at a time.

    For small numbers of inputs that fit in one batch,
    directly use `__call__()` for faster execution, e.g.,
    `model(x)`, or `model(x, training=False)` if you have layers such as
    `tf.keras.layers.BatchNormalization` that behave differently during
    inference. You may pair the individual model call with a `tf.function`
    for additional performance inside your inner loop.
    If you need access to numpy array values instead of tensors after your
    model call, you can use `tensor.numpy()` to get the numpy array value of
    an eager tensor.

    Also, note the fact that test loss is not affected by
    regularization layers like noise and dropout.

    Note: See [this FAQ entry](
    https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
    for more details about the difference between `Model` methods
    `predict()` and `__call__()`.

    Args:
        x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
          A more detailed description of unpacking behavior for iterator
          types (Dataset, generator, Sequence) is given in the `Unpacking
          behavior for iterator-like inputs` section of `Model.fit`.
        batch_size: Integer or `None`.
            Number of samples per batch.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of dataset, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = single line.
            `"auto"` becomes 1 for most cases, and to 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so `verbose=2` is
            recommended when not running interactively (e.g. in a production
            environment). Defaults to 'auto'.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`. If x is a `tf.data`
            dataset and `steps` is None, `predict()` will
            run until the input dataset is exhausted.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during prediction.
            See [callbacks](
            https://www.tensorflow.org/api_docs/python/tf/tf_keras/callbacks).
        max_queue_size: Integer. Used for generator or
            `keras.utils.Sequence` input only. Maximum size for the
            generator queue. If unspecified, `max_queue_size` will default
            to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-pickleable arguments to
            the generator as they can't be passed easily to children
            processes.

    See the discussion of `Unpacking behavior for iterator-like inputs` for
    `Model.fit`. Note that Model.predict uses the same interpretation rules
    as `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for
    all three methods.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        RuntimeError: If `model.predict` is wrapped in a `tf.function`.
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    version_utils.disallow_legacy_graph("Model", "predict")
    self._check_call_args("predict")
    training_module._disallow_inside_tf_function("predict")

    # TODO(yashkatariya): Cache model on the coordinator for faster
    # prediction.  If running under PSS, then swap it with OneDeviceStrategy
    # so that execution will run on the coordinator.
    original_pss_strategy = None
    if self.distribute_strategy._should_use_with_coordinator:
      original_pss_strategy = self.distribute_strategy
      self._distribution_strategy = None

    # Cluster coordinator is set by `.fit()` and `.evaluate()` which is not
    # needed in `.predict()` because all the predictions happen on the
    # coordinator/locally.
    if self._cluster_coordinator:
      self._cluster_coordinator = None

    verbose = training_module._get_verbosity(verbose, self.distribute_strategy)
    outputs = None
    with self.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      dataset_types = (tf.compat.v1.data.Dataset, tf.data.Dataset)
      if (self._in_multi_worker_mode() or training_module._is_tpu_multi_host(self.distribute_strategy)) and isinstance(
        x, dataset_types
      ):
        try:
          options = tf.data.Options()
          data_option = tf.data.experimental.AutoShardPolicy.DATA
          options.experimental_distribute.auto_shard_policy = data_option
          x = x.with_options(options)
        except ValueError:
          warnings.warn(
            "Using Model.predict with MultiWorkerMirroredStrategy "
            "or TPUStrategy and AutoShardPolicy.FILE might lead to "
            "out-of-order result. Consider setting it to "
            "AutoShardPolicy.DATA.",
            stacklevel=2,
          )

      data_handler = data_adapter.get_data_handler(
        x=x,
        batch_size=batch_size,
        steps_per_epoch=steps,
        initial_epoch=0,
        epochs=1,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        model=self,
        steps_per_execution=self._steps_per_execution,
      )

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
          callbacks,
          add_history=True,
          add_progbar=verbose != 0,
          model=self,
          verbose=verbose,
          epochs=1,
          steps=data_handler.inferred_steps,
        )

      self.predict_function = self.make_predict_function()
      self._predict_counter.assign(0)
      callbacks.on_predict_begin()
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.start()
      batch_outputs = None
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            tmp_batch_outputs = self.predict_function(iterator)
            if data_handler.should_sync:
              context.async_wait()
            batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
            if outputs is None:
              outputs = tf.nest.map_structure(
                lambda batch_output: [batch_output],
                batch_outputs,
              )
            else:
              tf.__internal__.nest.map_structure_up_to(
                batch_outputs,
                lambda output, batch_output: output.append(batch_output),
                outputs,
                batch_outputs,
              )
            end_step = step + data_handler.step_increment
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
      if batch_outputs is None:
        raise ValueError(
          "Unexpected result of `predict_function` "
          "(Empty batch_outputs). Please use "
          "`Model.compile(..., run_eagerly=True)`, or "
          "`tf.config.run_functions_eagerly(True)` for more "
          "information of where went wrong, or file a "
          "issue/bug to `tf.keras`."
        )
      if self.autotune_steps_per_execution:
        self._steps_per_execution_tuner.stop()
      callbacks.on_predict_end()
    all_outputs = tf.__internal__.nest.map_structure_up_to(
      batch_outputs, training_module.potentially_ragged_concat, outputs
    )

    # If originally PSS strategy was used, then replace it back since
    # predict is running under `OneDeviceStrategy` after the swap and once
    # its done we need to replace it back to PSS again.
    if original_pss_strategy is not None:
      self._distribution_strategy = original_pss_strategy

    return tf_utils.sync_to_numpy_or_python_type(all_outputs)

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

  def train_on_batch(
    self,
    x,
    y=None,
    sample_weight=None,
    class_weight=None,
    reset_metrics=True,
    return_dict=False,
  ):
    """Runs a single gradient update on a single batch of data.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
              (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case
          of temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample.
        class_weight: Optional dictionary mapping class indices (integers)
          to a weight (float) to apply to the model's loss for the samples
          from this class during training. This can be useful to tell the
          model to "pay more attention" to samples from an under-represented
          class. When `class_weight` is specified and targets have a rank of
          2 or greater, either `y` must be one-hot encoded, or an explicit
          final dimension of `1` must be included for sparse class labels.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated
          across batches.
        return_dict: If `True`, loss and metric results are returned as a
          dict, with each key being the name of the metric. If `False`, they
          are returned as a list.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
      RuntimeError: If `model.train_on_batch` is wrapped in a `tf.function`.
    """
    self._assert_compile_was_called()
    self._check_call_args("train_on_batch")
    training_module._disallow_inside_tf_function("train_on_batch")
    if reset_metrics:
      self.reset_metrics()
    with (
      self.distribute_strategy.scope(),
      training_utils.RespectCompiledTrainableState(  # noqa: E501
        self
      ),
    ):
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x, y, sample_weight, class_weight)
      self.train_function = self.make_train_function()
      logs = self.train_function(iterator)

    logs = tf_utils.sync_to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      return training_module.flatten_metrics_in_order(logs, self.metrics_names)

  def test_on_batch(
    self,
    x,
    y=None,
    sample_weight=None,
    reset_metrics=True,
    return_dict=False,
  ):
    """Test the model on a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays (in case the
              model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors (in case the model has
              multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case
          of temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated
          across batches.
        return_dict: If `True`, loss and metric results are returned as a
          dict, with each key being the name of the metric. If `False`, they
          are returned as a list.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: If `model.test_on_batch` is wrapped in a
          `tf.function`.
    """
    self._assert_compile_was_called()
    self._check_call_args("test_on_batch")
    training_module._disallow_inside_tf_function("test_on_batch")
    if reset_metrics:
      self.reset_metrics()
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x, y, sample_weight)
      self.test_function = self.make_test_function()
      logs = self.test_function(iterator)

    logs = tf_utils.sync_to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      return training_module.flatten_metrics_in_order(logs, self.metrics_names)

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays (in case the
              model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors (in case the model has
              multiple inputs).

    Returns:
        Numpy array(s) of predictions.

    Raises:
        RuntimeError: If `model.predict_on_batch` is wrapped in a
          `tf.function`.
    """
    self._check_call_args("predict_on_batch")
    training_module._disallow_inside_tf_function("predict_on_batch")
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x)
      self.predict_function = self.make_predict_function()
      outputs = self.predict_function(iterator)
    return tf_utils.sync_to_numpy_or_python_type(outputs)

  @doc_controls.do_not_generate_docs
  def fit_generator(
    self,
    generator,
    steps_per_epoch=None,
    epochs=1,
    verbose=1,
    callbacks=None,
    validation_data=None,
    validation_steps=None,
    validation_freq=1,
    class_weight=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0,
  ):
    """Fits the model on data yielded batch-by-batch by a Python generator.

    DEPRECATED:
      `Model.fit` now supports generators, so there is no longer any need to
      use this endpoint.
    """
    warnings.warn(
      "`Model.fit_generator` is deprecated and "
      "will be removed in a future version. "
      "Please use `Model.fit`, which supports generators.",
      stacklevel=2,
    )
    return self.fit(
      generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      verbose=verbose,
      callbacks=callbacks,
      validation_data=validation_data,
      validation_steps=validation_steps,
      validation_freq=validation_freq,
      class_weight=class_weight,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing,
      shuffle=shuffle,
      initial_epoch=initial_epoch,
    )

  @doc_controls.do_not_generate_docs
  def evaluate_generator(
    self,
    generator,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=0,
  ):
    """Evaluates the model on a data generator.

    DEPRECATED:
      `Model.evaluate` now supports generators, so there is no longer any
      need to use this endpoint.
    """
    warnings.warn(
      "`Model.evaluate_generator` is deprecated and "
      "will be removed in a future version. "
      "Please use `Model.evaluate`, which supports generators.",
      stacklevel=2,
    )
    self._check_call_args("evaluate_generator")

    return self.evaluate(
      generator,
      steps=steps,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing,
      verbose=verbose,
      callbacks=callbacks,
    )

  @doc_controls.do_not_generate_docs
  def predict_generator(
    self,
    generator,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=0,
  ):
    """Generates predictions for the input samples from a data generator.

    DEPRECATED:
      `Model.predict` now supports generators, so there is no longer any
      need to use this endpoint.
    """
    warnings.warn(
      "`Model.predict_generator` is deprecated and "
      "will be removed in a future version. "
      "Please use `Model.predict`, which supports generators.",
      stacklevel=2,
    )
    return self.predict(
      generator,
      steps=steps,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing,
      verbose=verbose,
      callbacks=callbacks,
    )

  def _check_call_args(self, method_name):
    """Check that `call()` has only one positional arg."""
    # Always allow first arg, regardless of arg name.
    fullargspec = self.main_model._call_spec.full_argspec
    if fullargspec.defaults:
      positional_args = fullargspec.args[: -len(fullargspec.defaults)]
    else:
      positional_args = fullargspec.args
    if "training" in positional_args:
      positional_args.remove("training")

    # self and first arg can be positional.
    if len(positional_args) > 2:
      extra_args = positional_args[2:]
      raise ValueError(
        f"Models passed to `{method_name}` can only have `training` "
        "and the first argument in `call()` as positional arguments, "
        f"found: {extra_args}."
      )

  def _validate_compile(self, optimizer, metrics, **kwargs):
    """Performs validation checks for the default `compile()`."""
    if any(isinstance(opt, optimizer_v1.Optimizer) for opt in tf.nest.flatten(optimizer)):
      raise ValueError(
        f"`tf.compat.v1.keras` Optimizer ({optimizer}) is "
        "not supported when eager execution is enabled. Use a "
        "`tf.keras` Optimizer instead, or disable eager "
        "execution."
      )

    kwargs.pop("cloning", None)  # Legacy DistStrat argument, never used.
    kwargs.pop("experimental_run_tf_function", None)  # Always `True`.
    distribute_arg = kwargs.pop("distribute", None)
    if distribute_arg is not None:
      raise ValueError(
        "`distribute` argument in compile is not available in TF 2.0. "
        "Please create the model under the `strategy.scope()`. "
        f"Received: {distribute_arg}."
      )
    target_tensor_arg = kwargs.pop("target_tensors", None)
    if target_tensor_arg is not None:
      raise ValueError(
        f"`target_tensors` argument is not supported when executing eagerly. Received: {target_tensor_arg}."
      )
    invalid_kwargs = set(kwargs) - {"sample_weight_mode"}
    if invalid_kwargs:
      raise TypeError(
        "Invalid keyword argument(s) in `compile()`: "
        f"{(invalid_kwargs,)}. Valid keyword arguments include "
        '"cloning", "experimental_run_tf_function", "distribute",'
        ' "target_tensors", or "sample_weight_mode".'
      )

    # Model must be created and compiled with the same DistStrat.
    if tf.distribute.has_strategy():
      strategy = tf.distribute.get_strategy()
      for v in self.main_model.variables:
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
            f"Variable ({v}) was not created in the distribution "
            f"strategy scope of ({strategy}). It is most likely "
            "because some layers, model, or optimizer was being "
            "created outside the distribution strategy scope. Try "
            "to make sure your code looks similar "
            "to the following.\nwith strategy.scope():\n"
            "  model=_create_model()\n"
            "  model.compile(...)"
          )

    # Model metrics must be created in the same distribution strategy scope
    # as the model.
    strategy = self.distribute_strategy
    for metric in tf.nest.flatten(metrics):
      for v in getattr(metric, "variables", []):
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
            f"Metric ({metric}) passed to `model.compile` was "
            "created inside a different distribution strategy "
            "scope than the model. All metrics must be created "
            "in the same distribution strategy "
            f"scope as the model (in this case {strategy}). "
            "If you pass in a string identifier for a metric to "
            "compile, the metric will automatically be created "
            "in the correct distribution strategy scope."
          )

    # Model metrics must be created in the same distribution strategy scope
    # as the model.
    for opt in tf.nest.flatten(optimizer):
      for v in getattr(opt, "_weights", []):
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
            f"Optimizer ({optimizer}) passed to `model.compile` "
            "was created inside a different distribution strategy "
            "scope than the model. All optimizers must be created "
            "in the same distribution strategy scope as the model "
            f"(in this case {strategy}). If you pass in a string "
            "identifier for an optimizer to compile, the optimizer "
            "will automatically be created in the correct "
            "distribution strategy scope."
          )

  def _maybe_load_initial_counters_from_ckpt(self, steps_per_epoch, initial_epoch):
    """Maybe load initial epoch from ckpt, considering worker recovery.

    Refer to tensorflow/python/tf_keras/distribute/worker_training_state.py
    for more information.

    Args:
      steps_per_epoch: The number of step per epoch.
      initial_epoch: The original initial_epoch user passes in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the (epoch, step) the training is supposed to
      continue at. Otherwise, return the `initial_epoch, initial_step` the
      user passes in.
    """
    initial_step = 0
    if self._training_state is not None:
      return self._training_state.maybe_load_initial_counters_from_ckpt(
        steps_per_epoch, initial_epoch, mode=ModeKeys.TRAIN
      )
    return (initial_epoch, initial_step)

  def _assert_compile_was_called(self):
    # Checks whether `compile` has been called. If it has been called,
    # then the optimizer is set. This is different from whether the
    # model is compiled
    # (i.e. whether the model is built and its inputs/outputs are set).
    if not self._is_compiled:
      raise RuntimeError("You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.")

  def _check_sample_weight_warning(self, x, sample_weight):
    # Datasets can include sample weight, by returning a tuple with the
    # structure of `(x, y, sample_weight)`.
    sample_weight_present = sample_weight is not None or (
      isinstance(x, tf.data.Dataset) and isinstance(x.element_spec, tuple) and len(x.element_spec) == 3
    )

    if sample_weight_present and self.compiled_metrics._user_weighted_metrics is None:
      logging.warning(
        "`evaluate()` received a value for `sample_weight`, but "
        "`weighted_metrics` were not provided.  Did you mean to pass "
        "metrics to `weighted_metrics` in `compile()`?  If this is "
        "intentional you can pass `weighted_metrics=[]` to `compile()` "
        "in order to silence this warning."
      )

  def _should_eval(self, epoch, validation_freq):
    epoch = epoch + 1  # one-index the user-facing epoch.
    if isinstance(validation_freq, int):
      return epoch % validation_freq == 0
    elif isinstance(validation_freq, list):
      return epoch in validation_freq
    else:
      raise ValueError(
        "Expected `validation_freq` to be a list or int. "
        f"Received: validation_freq={validation_freq} of the "
        f"type {type(validation_freq)}."
      )

  ######################################################################
  # Functions below exist only as v1 / v2 compatibility shims.
  ######################################################################

  def _get_compile_args(self, user_metrics=True):
    """Used for saving or cloning a Model.

    Args:
      user_metrics: Whether to return user-supplied metrics or `Metric`
        objects. If True, returns the user-supplied metrics.
        Defaults to `True`.

    Returns:
      Dictionary of arguments that were used when compiling the model.
    """
    self._assert_compile_was_called()
    saved_metrics = self.compiled_metrics._user_metrics
    saved_weighted_metrics = self.compiled_metrics._user_weighted_metrics

    if not user_metrics:
      if saved_metrics is not None:
        saved_metrics = self.compiled_metrics._metrics
      if saved_weighted_metrics is not None:
        saved_weighted_metrics = self.compiled_metrics._weighted_metrics

    compile_args = {
      "optimizer": self.optimizer,
      "loss": self.compiled_loss._user_losses,
      "metrics": saved_metrics,
      "weighted_metrics": saved_weighted_metrics,
      "loss_weights": self.compiled_loss._user_loss_weights,
    }
    return compile_args

  def _get_callback_model(self):
    return self

  def _in_multi_worker_mode(self):
    return self.distribute_strategy.extended._in_multi_worker_mode()

  @property
  def _compile_was_called(self):
    return self._is_compiled

  @property
  def main_model(self):
    """
    Returns:
      The main model
    """
    if len(self._model) == 1:
      return self._model["main"]
    else:
      for name, _model in self._model.items():
        if "main" in name:
          return _model
      ValueError("Could not find the main model.")

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _maybe_create_attribute(self, name, default_value):
    """Create attribute (with the default value) if it hasn't been created.

    This is useful for fields that is used for tracking purpose,
    _trainable_weights, or _layers. Note that user could create a layer
    subclass and assign an internal field before invoking the
    Layer.__init__(), the __setattr__() need to create the tracking fields
    and __init__() need to not override them.

    Args:
      name: String, the name of the attribute.
      default_value: Object, the default value of the attribute.
    """
    if not hasattr(self, name):
      self.__setattr__(name, default_value)

  def _get_trainable_state(self):
    """Get the `trainable` state of each sublayer.

    Returns:
      A dict mapping all sublayers to their `trainable` value.
    """
    trainable_state = weakref.WeakKeyDictionary()
    for layer in self.main_model._flatten_layers():
      trainable_state[layer] = layer.trainable
    return trainable_state
