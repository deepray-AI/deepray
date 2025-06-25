# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import weakref

import tensorflow as tf
from absl import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.resource_variable_ops import (
  get_eager_safe_handle_data,
  _combine_handle_data,
  _set_handle_shapes_and_types,
  ResourceVariable,
)
from tensorflow.python.platform import resource_loader
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import compat

from deepray.custom_ops.embedding_variable import config_pb2
from deepray.custom_ops.embedding_variable import variables as ev_variables
from deepray.utils import logging_util

gen_kv_variable_ops = tf.load_op_library(resource_loader.get_path_to_datafile("../_kv_variable_ops.so"))

logger = logging_util.get_logger()

__all__ = ["EmbeddingVariable"]


def _variable_handle_from_shape_and_dtype(shape, dtype, key_type, shared_name, name, graph_mode, initial_value=None):
  """Create a variable handle, copying in handle data from `initial_value`."""
  container = ops.get_default_graph()._container  # pylint: disable=protected-access
  if container is None:
    container = ""
  shape = tensor_shape.as_shape(shape)
  dtype = dtypes.as_dtype(dtype)
  key_type = dtypes.as_dtype(key_type)

  handle = gen_kv_variable_ops.kv_var_handle_op(
    shape=shape,
    dtype=dtype,
    Tkeys=key_type,
    shared_name=shared_name,
    # debug_name=name,
    name=name,
    container=container,
  )
  if initial_value is None:
    initial_value = handle
  if graph_mode:
    full_handle_data = _combine_handle_data(handle, initial_value)
    _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
    return handle
  else:
    handle_data = handle_data_util.create_handle_data(shape, dtype)
    if initial_value is not None and initial_value.dtype == dtypes.variant:
      extra_handle_data = get_eager_safe_handle_data(initial_value)
      if extra_handle_data is not None and extra_handle_data.is_set:
        if not handle_data.is_set or len(handle_data.shape_and_type) != 1:
          raise RuntimeError(f"Expected VarHandleOp to return a length==1 shape_and_type, but saw: '{handle_data}'")
        handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)

    _set_handle_shapes_and_types(handle, handle_data, graph_mode)
    return handle


def eager_safe_variable_handle(initial_value, shape, key_type, shared_name, name, graph_mode):
  """Creates a variable handle with information to do shape inference.

  The dtype is read from `initial_value` and stored in the returned
  resource tensor's handle data.

  If `initial_value.dtype == tf.variant`, we additionally extract the handle
  data (if any) from `initial_value` and append it to the `handle_data`.
  In this case, the returned tensor's handle data is in the form

  ```
  is_set: true
  shape_and_type {
    shape {
      // initial_value.shape
    }
    dtype: DT_VARIANT
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[0]
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[1]
  }
  ...
  ```

  Ops that read from this tensor, such as `ReadVariableOp` and
  `AssignVariableOp`, know that `handle_data(handle).shape_and_type[1:]`
  correspond to the handle data of the variant(s) stored in the Variable.

  Args:
    initial_value: A `Tensor`.
    shape: The shape of the handle data. Can be `TensorShape(None)` (i.e.
      unknown shape).
    shared_name: A string.
    name: A string.
    graph_mode: A python bool.

  Returns:
    The handle, a `Tensor` of type `resource`.
  """
  dtype = initial_value.dtype.base_dtype
  return _variable_handle_from_shape_and_dtype(shape, dtype, key_type, shared_name, name, graph_mode, initial_value)


class EmbeddingVariable(ResourceVariable, saveable_object.SaveableObject):
  """Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
  ```
  """

  def __init__(
    self,  # pylint: disable=super-init-not-called
    initial_value=None,
    trainable=None,
    collections=None,
    validate_shape=True,  # pylint: disable=unused-argument
    caching_device=None,
    name=None,
    dtype=None,
    variable_def=None,
    import_scope=None,
    constraint=None,
    distribute_strategy=None,
    synchronization=None,
    aggregation=None,
    shape=None,
    handle=None,
    experimental_enable_variable_lifting=None,
    invalid_key=None,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. (Note that
        initializer functions from init_ops.py must first be bound to a shape
        before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      handle: (optional) The handle of a `tf.Variable`. If provided, only
        `trainable`, `shape`, `dtype`, and `handle` will be used to construct
        this `tf.Variable`.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError(
          f"The variable_def and initial_value args to "
          f"`tf.Variable` are mutually exclusive, but got both: "
          f"variable_def={variable_def},\n"
          f"initial_value={initial_value}"
        )
      if context.executing_eagerly():
        raise ValueError(
          f"Creating a `tf.Variable` with a `variable_def` arg "
          f"is not supported when eager execution is enabled. "
          f"Got: variable_def={variable_def}"
        )
      self._init_from_proto(variable_def, import_scope=import_scope, validate_shape=validate_shape)
    elif handle is not None:
      self._init_from_handle(trainable=trainable, shape=shape, dtype=dtype, handle=handle)
    else:
      evconfig.reveal()
      self._init_from_args(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        name=name,
        dtype=dtype,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
        distribute_strategy=distribute_strategy,
        validate_shape=validate_shape,
        experimental_enable_variable_lifting=experimental_enable_variable_lifting,
        invalid_key=invalid_key,
        evconfig=evconfig,
        ht_partition_num=ht_partition_num,
      )

  def __repr__(self):
    return "<tf.EmbeddingVariable '%s' embedding dim=%s dtype=%s>" % (self.name, self.shape, self.dtype.name)

  def _init_from_args(
    self,
    initial_value=None,
    trainable=None,
    collections=None,
    caching_device=None,
    name=None,
    dtype=None,
    constraint=None,
    synchronization=None,
    aggregation=None,
    distribute_strategy=None,
    shape=None,
    validate_shape=True,
    experimental_enable_variable_lifting=None,
    invalid_key=-1,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable was
        created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
    synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(
      synchronization, aggregation, trainable, name
    )
    if experimental_enable_variable_lifting is None:
      experimental_enable_variable_lifting = True
    if initial_value is None:
      raise ValueError(
        "The `initial_value` arg to `tf.Variable` must "
        "be specified except when you are not providing a "
        "`variable_def`. You provided neither."
      )
    init_from_fn = callable(initial_value)

    if (
      isinstance(initial_value, tensor_module.Tensor)
      and hasattr(initial_value, "graph")
      and initial_value.graph.building_function
    ):
      raise ValueError(
        f"Argument `initial_value` ({initial_value}) could not "
        "be lifted out of a `tf.function`. "
        f"(Tried to create variable with name='{name}'). "
        "To avoid this error, when constructing `tf.Variable`s "
        "inside of `tf.function` you can create the "
        "`initial_value` tensor in a "
        "`tf.init_scope` or pass a callable `initial_value` "
        "(e.g., `tf.Variable(lambda : "
        "tf.truncated_normal([10, 40]))`). "
        "Please file a feature request if this "
        "restriction inconveniences you."
      )

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
        f"collections argument to Variable constructor must be a list, "
        f"tuple, or set. Got {collections} of type {type(collections)}"
      )
    if constraint is not None and not callable(constraint):
      raise ValueError(
        f"Argument `constraint` must be None or a callable. a callable. Got a {type(constraint)}:  {constraint}"
      )

    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

    self._save_slice_info = None
    self._in_graph_mode = not context.executing_eagerly()
    self._steps_to_live = evconfig.steps_to_live
    self._init_data_source = evconfig.init_data_source
    self._emb_index = evconfig.emb_index
    self._slot_index = evconfig.slot_index
    self._block_num = evconfig.block_num
    self._block_handle_name = None
    self._primary = evconfig.primary
    self._ht_type = evconfig.ht_type
    self._ht_partition_num = ht_partition_num
    self._is_sparse = False
    self.importer = None
    if evconfig.filter_strategy != None:
      if isinstance(evconfig.filter_strategy, ev_variables.CounterFilter):
        self._filter_freq = evconfig.filter_strategy.filter_freq
        self._max_element_size = 0
        self._false_positive_probability = -1.0
        self._counter_type = dtypes.uint64
      elif isinstance(evconfig.filter_strategy, ev_variables.CBFFilter):
        self._filter_freq = evconfig.filter_strategy.filter_freq
        self._max_element_size = evconfig.filter_strategy.max_element_size
        self._false_positive_probability = evconfig.filter_strategy.false_positive_probability
        self._counter_type = evconfig.filter_strategy.counter_type
    else:
      self._filter_freq = 0
      self._max_element_size = 0
      self._false_positive_probability = -1.0
      self._counter_type = dtypes.uint64

    self._record_freq = os.environ.get("TF_RECORD_FREQ", "0") == "1"
    self._record_version = os.environ.get("TF_RECORD_VERSION", "0") == "1"
    self._l2_weight_threshold = evconfig.l2_weight_threshold
    self._storage_type = evconfig.storage_type
    self._storage_path = evconfig.storage_path
    self._storage_size = evconfig.storage_size
    self._default_value_dim = evconfig.default_value_dim
    self._default_value_no_permission = evconfig.default_value_no_permission
    self._storage_cache_strategy = evconfig.storage_cache_strategy
    self._layout = evconfig.layout

    if self._primary is None:
      self._is_primary = True
    else:
      self._is_primary = False

    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if experimental_enable_variable_lifting:
      maybe_init_scope = ops.init_scope
    else:
      maybe_init_scope = contextlib.nullcontext
    with maybe_init_scope():
      with ops.name_scope(name, "Variable", [] if init_from_fn else [initial_value], skip_on_eager=False) as name:
        self._invalid_key = invalid_key
        self._invalid_key_type = ops.convert_to_tensor(invalid_key, name="invalid_key").dtype.base_dtype
        handle_name = ops.name_from_scope_name(name)
        shared_name = handle_name
        if self._in_graph_mode:
          unique_id = shared_name
        else:
          # When in eager mode, use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
        self._unique_id = unique_id
        if handle_name is None:
          self._handle_name = "Variable:0"
        else:
          self._handle_name = handle_name + ":0"
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = ops.device if self._in_graph_mode else ops.NullContextmanager
        attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=[compat.as_bytes("loc:@%s" % handle_name)])
        )
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            if init_from_fn:
              initial_value = initial_value()
            if isinstance(initial_value, trackable.CheckpointInitialValue):
              self._maybe_initialize_trackable()
              self._update_uid = initial_value.checkpoint_position.restore_uid
              initial_value = initial_value.wrapped_value
            initial_value = ops.convert_to_tensor(initial_value, name="initial_value", dtype=dtype)
            rank = initial_value.get_shape().rank - 1
          if shape is not None:
            if not initial_value.shape.is_compatible_with(shape):
              raise ValueError(
                f"In this `tf.Variable` creation, the initial value's shape "
                f"({initial_value.shape}) is not compatible with "
                f"the explicitly supplied `shape` argument ({shape})."
              )
          else:
            shape = initial_value.get_shape()[rank:]
          _device = (
            "GPU"
            if self._storage_type
            in [config_pb2.StorageType.HBM, config_pb2.StorageType.HBM_DRAM, config_pb2.StorageType.HBM_DRAM_SSDHASH]
            else "CPU"
          )
          with ops.device(_device):
            handle = eager_safe_variable_handle(
              initial_value=initial_value,
              shape=shape,
              key_type=self._invalid_key_type,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode,
            )
          handle._parent_trackable = weakref.ref(self)
          handle._name = handle_name + ":0"
          handle._unique_id = unique_id
          self._handle = handle
        # pylint: disable=protected-access
        if (
          self._in_graph_mode and initial_value is not None and initial_value.op._get_control_flow_context() is not None
        ):
          raise ValueError(
            f"The `initial_value` passed to `tf.Variable` {name} is from "
            f"inside a control-flow  construct, such as a loop or "
            f"conditional. When creating a "
            f"`tf.Variable` inside a loop or conditional, use a lambda as "
            f"the `initial_value`. Got: initial_value=({initial_value})"
          )
        # pylint: enable=protected-access
        dtype = initial_value.dtype.base_dtype
        self._counts_tensor = {}
        self._is_multi_tier = self.is_multi_tier(self._storage_type)
        if self._primary is None:
          self._primary = self

        if self._is_primary:
          self._slot_num = flags.FLAGS.ev_slot_num
        else:
          self._slot_num = evconfig.slot_num

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = gen_kv_variable_ops.kv_var_is_initialized_op(
              handle, Tkeys=self._invalid_key_type, dtype=self._dtype
            )
          if initial_value is not None:
            # pylint: disable=g-backslash-continuation
            with (
              ops.name_scope("Assign") as n,
              ops.colocate_with(None, ignore_existing=True),
              ops.device(handle.device),
            ):
              with ops.control_dependencies(None if self._is_primary else [self._primary.initializer]):
                self._init_op = gen_kv_variable_ops.initialize_kv_variable_v2_op(
                  handle,
                  self._primary._handle,
                  variables._try_guard_against_uninitialized_dependencies(name, initial_value),
                  ops.convert_to_tensor(invalid_key),
                  slot_num=self._slot_num,
                  shape=initial_value.get_shape()[rank:],
                  steps_to_live=self._steps_to_live,
                  emb_index=self._emb_index,
                  block_num=self.block_num,
                  slot_index=self._slot_index,
                  ht_type=self._ht_type,
                  ht_partition_num=self._ht_partition_num,
                  filter_freq=self._filter_freq,
                  l2_weight_threshold=self._l2_weight_threshold,
                  max_element_size=self._max_element_size,
                  false_positive_probability=self._false_positive_probability,
                  counter_type=self._counter_type,
                  max_freq=99999,
                  layout=self._layout,
                  storage_type=self._storage_type,
                  storage_path=self._storage_path,
                  storage_size=self._storage_size,
                  default_value_dim=self._default_value_dim,
                  default_value_no_permission=self._default_value_no_permission,
                  record_freq=self._record_freq,
                  record_version=self._record_version,
                  embedding_variable_type=config_pb2.EmbeddingVariableType.IMMUTABLE,
                  name=n,
                )
              set_attr_ops = []

              if self._is_primary and self._is_multi_tier:
                with ops.control_dependencies([self._init_op]):
                  set_cache_strategy_op = gen_kv_variable_ops.kv_resource_init_cache_strategy_op(
                    self._handle, cache_strategy=self._storage_cache_strategy, Tkeys=self._invalid_key_type, dtype=dtype
                  )
                set_attr_ops.append(set_cache_strategy_op)
              with ops.control_dependencies(set_attr_ops + [self._init_op]):
                self._initializer_op = control_flow_ops.no_op()

              self.create_init_op_for_restore(name, initial_value, invalid_key, rank)
        else:
          self._init_op = gen_kv_variable_ops.initialize_kv_variable_v2_op(
            handle,
            self._primary._handle,
            initial_value,
            ops.convert_to_tensor(invalid_key),
            slot_num=self._slot_num,
            shape=shape,
            steps_to_live=self._steps_to_live,
            emb_index=self._emb_index,
            block_num=self.block_num,
            slot_index=self._slot_index,
            ht_type=self._ht_type,
            ht_partition_num=self._ht_partition_num,
            filter_freq=self._filter_freq,
            l2_weight_threshold=self._l2_weight_threshold,
            max_element_size=self._max_element_size,
            false_positive_probability=self._false_positive_probability,
            counter_type=self._counter_type,
            max_freq=99999,
            layout=self._layout,
            storage_type=self._storage_type,
            storage_path=self._storage_path,
            storage_size=self._storage_size,
            default_value_dim=self._default_value_dim,
            default_value_no_permission=self._default_value_no_permission,
            record_freq=self._record_freq,
            record_version=self._record_version,
            embedding_variable_type=config_pb2.EmbeddingVariableType.IMMUTABLE,
          )
          if self._is_primary and self._is_multi_tier:
            with ops.control_dependencies([self._init_op]):
              set_cache_strategy_op = gen_kv_variable_ops.kv_resource_init_cache_strategy_op(
                self._handle, cache_strategy=self._storage_cache_strategy, Tkeys=self._invalid_key_type, dtype=dtype
              )

        if self._in_graph_mode:
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      super(EmbeddingVariable, self).__init__(
        trainable=trainable,
        shape=shape,
        dtype=dtype,
        handle=handle,
        synchronization=synchronization,
        constraint=constraint,
        aggregation=aggregation,
        distribute_strategy=distribute_strategy,
        name=name,
        initial_value=initial_value,
        caching_device=caching_device,
        validate_shape=validate_shape,
      )

  def is_multi_tier(self, storage_type):
    multi_level_list = [
      config_pb2.StorageType.LEVELDB,
      config_pb2.StorageType.SSDHASH,
      config_pb2.StorageType.DRAM_PMEM,
      config_pb2.StorageType.DRAM_LEVELDB,
      config_pb2.StorageType.DRAM_SSDHASH,
      config_pb2.StorageType.HBM_DRAM,
      config_pb2.StorageType.DRAM_PMEM_SSDHASH,
      config_pb2.StorageType.HBM_DRAM_SSDHASH,
    ]
    return storage_type in multi_level_list

  def create_init_op_for_restore(self, name, initial_value, invalid_key, rank):
    with ops.control_dependencies(None if self._is_primary else [self._primary._init_op_for_restore]):
      self._initializer_for_restore = gen_kv_variable_ops.initialize_kv_variable_v2_op(
        self._handle,
        self._primary._handle,
        variables._try_guard_against_uninitialized_dependencies(name, initial_value),
        ops.convert_to_tensor(invalid_key),
        initial_num_buckets=config_pb2.IsSetInitialized.NOT_SET_INITAILIZED,
        slot_num=self._slot_num,
        shape=initial_value.get_shape()[rank:],
        steps_to_live=self._steps_to_live,
        emb_index=self._emb_index,
        block_num=self.block_num,
        slot_index=self._slot_index,
        ht_type=self._ht_type,
        ht_partition_num=self._ht_partition_num,
        filter_freq=self._filter_freq,
        l2_weight_threshold=self._l2_weight_threshold,
        max_element_size=self._max_element_size,
        false_positive_probability=self._false_positive_probability,
        counter_type=self._counter_type,
        max_freq=99999,
        layout=self._layout,
        storage_type=self._storage_type,
        storage_path=self._storage_path,
        storage_size=self._storage_size,
        default_value_dim=self._default_value_dim,
        default_value_no_permission=self._default_value_no_permission,
        record_freq=self._record_freq,
        record_version=self._record_version,
        embedding_variable_type=config_pb2.EmbeddingVariableType.IMMUTABLE,
      )
    set_attr_ops = []
    if self._is_primary and self._is_multi_tier:
      with ops.control_dependencies([self._initializer_for_restore]):
        set_cache_op = gen_kv_variable_ops.kv_resource_init_cache_strategy_op(
          self._handle, cache_strategy=self._storage_cache_strategy, Tkeys=self._invalid_key_type, dtype=self._dtype
        )
      set_attr_ops.append(set_cache_op)
    with ops.control_dependencies(set_attr_ops + [self._initializer_for_restore]):
      self._init_op_for_restore = control_flow_ops.no_op()
    # self.collect_restore_denpendencies()

  def sparse_read(self, indices, name=None, ev_init_value=None, counts=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      if self._trainable:
        tape.variable_accessed(self)
      if ev_init_value is not None:
        default_value = math_ops.cast(ev_init_value, self.dtype)
        is_use_default_value_tensor = True
      else:
        default_value = ops.convert_to_tensor(1.0, dtype=self.dtype)
        is_use_default_value_tensor = False
      if counts is not None:
        value = gen_kv_variable_ops.kv_resource_gather_v1(
          self._handle, indices, default_value, counts, is_inference=True, name=name
        )
        self._counts_tensor[indices] = counts
      else:
        value = gen_kv_variable_ops.kv_resource_gather(
          self._handle, indices, default_value, is_use_default_value_tensor, is_inference=True, name=name
        )
    return value

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initializer_op

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable."""
    if context.executing_eagerly():
      raise RuntimeError("initial_value not supported in EAGER mode.")
    return self._initial_value

  def is_initialized(self):
    return gen_kv_variable_ops.kv_var_is_initialized_op(self._handle, Tkeys=self._invalid_key_type, dtype=self._dtype)

  def is_all_slot_initialized(self):
    return gen_kv_variable_ops.kv_var_is_all_slot_initialized_op(
      self._handle, Tkeys=self._invalid_key_type, dtype=self._dtype
    )

  @property
  def block_num(self):
    if self._block_num is None:
      return 1
    else:
      return self._block_num

  def need_counts(self):
    return self._record_freq or (self._filter_freq > 0) or self._is_multi_tier

  @property
  def storage_type(self):
    return self._storage_type

  def lookup_resource(self):
    return gen_kv_variable_ops.kv_resource_lookup_resource(self.handle, Tkeys=self._invalid_key_type, dtype=self._dtype)

  # Unused
  # def _gather_saveables_for_checkpoint(self):
  #   return {"foo": lambda name: EmbeddingVariableSaveable(self, name)}


def lookup_resource(var):
  return gen_kv_variable_ops.kv_resource_lookup_resource(var.handle, Tkeys=var._invalid_key_type, dtype=var._dtype)


def variable_shape(handle, indices, grad):
  handle_data = get_eager_safe_handle_data(handle)
  if handle_data is None or not handle_data.is_set:
    return gen_kv_variable_ops.kv_variable_shape(handle, Tkeys=indices.dtype, dtype=grad.dtype)
  shape_proto = handle_data.shape_and_type[0].shape
  if shape_proto.unknown_rank or any(x.size == -1 for x in shape_proto.dim):
    return gen_kv_variable_ops.kv_variable_shape(handle, Tkeys=indices.dtype, dtype=grad.dtype)
  return constant_op.constant([x.size for x in shape_proto.dim], dtype=dtypes.int32)


def get_tensor_slices(trackables):
  tensor_names = []
  shapes_and_slices = []
  tensors = []
  restored_trackables = []
  ev_names = []
  ev_resources = []
  ev_key_types = []
  has_ev = False
  for obj_prefix, obj in trackables.items():
    if isinstance(obj, EmbeddingVariable):
      ev_names.append(obj.name)
      ev_resources.append(obj.lookup_resource())
      ev_key_types.append(obj._invalid_key_type)
      has_ev = True

    tensor_names.append(obj_prefix + "/value")
    shapes_and_slices.append("")
    tensors.append(constant_op.constant(2, dtype=obj.dtype))
  return tensor_names, shapes_and_slices, tensors, restored_trackables, ev_names, ev_resources, ev_key_types, has_ev


def save_fn(trackables, file_prefix):
  """Save stack and part objects to a checkpoint shard."""
  tensor_names, shapes_and_slices, tensors, _, ev_names, ev_resources, ev_key_types, has_ev = get_tensor_slices(
    trackables
  )
  gen_kv_variable_ops.save_v3(
    file_prefix, tensor_names, shapes_and_slices, ev_names, ev_resources, tensors, ev_key_types, has_ev
  )
  return file_prefix


restore_queue = dict()


def restore_fn(trackables, merged_prefix):
  for obj_prefix, obj in trackables.items():
    # Initialize queue entry if not exists
    if obj._primary.name not in restore_queue:
      restore_queue[obj._primary.name] = []
    restore_queue[obj._primary.name].append(obj)
    if obj.is_all_slot_initialized():
      for ev in restore_queue[obj._primary.name]:
        gen_kv_variable_ops.kv_resource_import_v3(
          merged_prefix,
          ev.handle,
          ev.name,
          ops.convert_to_tensor(ev._invalid_key),
          shape=ev.shape,
          partition_id=0,
          partition_num=1,
          dtype=ev.dtype,
        )


registration.register_checkpoint_saver(
  name="EmbeddingVariable",
  predicate=lambda x: isinstance(x, (EmbeddingVariable)),
  save_fn=save_fn,
  restore_fn=restore_fn,
)


@ops.RegisterGradient("KvResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = variable_shape(handle, indices, grad)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[0:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [indexed_slices.IndexedSlices(values, indices, params_shape), None, None]


@ops.RegisterGradient("KvResourceGatherV1")
def _GatherV1Grad(op: ops.Operation, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = variable_shape(handle, indices, grad)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[0:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [indexed_slices.IndexedSlices(values, indices, params_shape), None, None]


ops.NotDifferentiable("KvVarIsInitializedOp")
ops.NotDifferentiable("KvVariableShape")


class EmbeddingVariableSaveable(saveable_object.SaveableObject):
  """SaveableObject implementation that handles EmbeddingVariables."""

  def __init__(self, var, name):
    self.handle_op = var.handle
    self.invalid_key = var.invalid_key
    self.dtype = var._dtype
    self.key_type = var._invalid_key_type
    self.steps_to_live = var.steps_to_live
    self.ht_type = var._ht_type
    self.ht_partition_num = var._ht_partition_num
    name = var._shared_name
    self.var = var
    is_partitioned_ev = not isinstance(self.var._save_slice_info, str)
    self.partition_id = 0
    self.partition_num = 1
    if self.var._save_slice_info is not None:
      self.partition_id = self.var._save_slice_info.var_offset[0] if is_partitioned_ev else 0
      self.partition_num = self.var._save_slice_info.full_shape[0] if is_partitioned_ev else 1

    def _read_variable_closure(v):
      def f():
        with ops.device(v.device):
          x = v.read_value()
          return array_ops.identity(x)

      return f

    unused_tensor = var.handle
    self.resource = lookup_resource(var)

    specs = []
    specs.append(saveable_object.SaveSpec(unused_tensor, "", name + "-keys", dtype=self.key_type, device=var.device))
    specs.append(saveable_object.SaveSpec(unused_tensor, "", name + "-values", dtype=dtypes.float32, device=var.device))
    specs.append(saveable_object.SaveSpec(unused_tensor, "", name + "-versions", dtype=dtypes.int64, device=var.device))
    specs.append(saveable_object.SaveSpec(unused_tensor, "", name + "-freqs", dtype=dtypes.int64, device=var.device))

    # pylint: disable=protected-access
    super(EmbeddingVariableSaveable, self).__init__(var, specs, name)
    self.is_sparse = var._is_sparse

  def restore(self, restored_tensors, unused_restored_shapes):
    # pylint: disable=protected-access
    with ops.device("/cpu:0"):
      name_tensor = ops.convert_to_tensor(self.name)
    with ops.colocate_with(self.handle_op):
      handle_name = ops.name_from_scope_name(self.name)
      is_partitioned_ev = not isinstance(self.var._save_slice_info, str)
      if self.var._init_data_source is not None:
        return self.var.recover_from_init_data_source(self.var._init_data_source, self.partition_id, self.partition_num)
      else:
        restore_dependency = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLE_RESTORE_DEPENDENCY)[0]
        with ops.control_dependencies(restore_dependency[self.var._primary_handle]):
          rank = self.op.initial_value.get_shape().rank - 1
          restore_op = gen_kv_variable_ops.kv_resource_import_v3(
            restored_tensors[0],
            self.handle_op,
            name_tensor,
            ops.convert_to_tensor(self.invalid_key),
            shape=self.op.initial_value.get_shape()[rank:],
            partition_id=self.partition_id,
            partition_num=self.partition_num,
            dtype=self.var._dtype,
          )
        return restore_op

  def incr_restore(self, restored_tensors, unused_restored_shapes):
    # pylint: disable=protected-access
    name_tensor = ops.convert_to_tensor(self.name)
    with ops.colocate_with(self.handle_op):
      handle_name = ops.name_from_scope_name(self.name)
      return gen_kv_variable_ops.kv_resource_incr_import(
        restored_tensors[0],
        self.handle_op,
        name_tensor,
        ops.convert_to_tensor(self.invalid_key),
        variables._try_guard_against_uninitialized_dependencies(self.name, self.op.initial_value),
        partition_id=self.partition_id,
        partition_num=self.partition_num,
      )
