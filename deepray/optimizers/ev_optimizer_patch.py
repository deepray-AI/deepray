# Copyright 2024 The Deepray Authors. All Rights Reserved.
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
"""EmbeddingVariable optimizer."""

import tensorflow as tf
from packaging.version import parse
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from deepray.custom_ops.embedding_variable import config_pb2
from deepray.custom_ops.embedding_variable import variables as ev_variables
from deepray.custom_ops.unique_ops import gen_array_ops

if parse(tf.__version__) < parse("2.11.0"):
  from keras.optimizers.legacy.optimizer_v2 import _var_key
elif parse(tf.__version__) > parse("2.16.0"):
  from tf_keras.src.optimizers.legacy.optimizer_v2 import _var_key
  from tf_keras.src import backend
  from tf_keras.src.optimizers.legacy.optimizer_v2 import _deduplicate_indexed_slices
else:
  from keras.src.optimizers.legacy.optimizer_v2 import _var_key
  from keras.src import backend
  from keras.src.optimizers.legacy.optimizer_v2 import _deduplicate_indexed_slices

import tf_keras as keras
import functools

from deepray.custom_ops.embedding_variable.python import kv_variable_ops

from tensorflow.core.framework import attr_value_pb2
from deepray.custom_ops.embedding_variable.variable_scope import (
  get_embedding_variable_internal,
  get_embedding_variable_v2_internal,
)


class SlotConfig:
  def __init__(self, slot_num=1, slot_index=0, slot_type=config_pb2.SlotType.EMBEDDING_VARIABLE):
    self.slot_num = slot_num
    self.slot_index = slot_index
    self.slot_type = slot_type


def _set_init_op_embedding_type_attr(var, embedding_type):
  var._init_op._set_attr("embedding_variable_type", attr_value_pb2.AttrValue(i=embedding_type))
  var._initializer_for_restore._set_attr("embedding_variable_type", attr_value_pb2.AttrValue(i=embedding_type))


def _set_init_op_slot_num_attr(var, slot_num):
  var._init_op._set_attr("slot_num", attr_value_pb2.AttrValue(i=slot_num))
  var._initializer_for_restore._set_attr("slot_num", attr_value_pb2.AttrValue(i=slot_num))


def add_slot(self, var, slot_name, initializer="zeros", shape=None, slot_config=None):
  """Add a new slot variable for `var`.

  A slot variable is an additional variable associated with `var` to
  train.  It is allocated and managed by optimizers, e.g. `Adam`.

  Args:
    var: a `Variable` object.
    slot_name: name of the slot variable.
    initializer: initializer of the slot variable
    shape: (Optional) shape of the slot variable. If not set, it will
      default to the shape of `var`.

  Returns:
    A slot variable.
  """
  if slot_name not in self._slot_names:
    self._slot_names.append(slot_name)
  var_key = _var_key(var)
  slot_dict = self._slots.setdefault(var_key, {})
  weight = slot_dict.get(slot_name, None)
  if weight is None:
    if isinstance(initializer, str) or callable(initializer):
      initializer = keras.initializers.get(initializer)
      if isinstance(
        initializer,
        tf.__internal__.tracking.CheckpointInitialValueCallable,
      ) or (shape is not None):
        slot_shape = shape
      else:
        slot_shape = var.shape
      initial_value = functools.partial(initializer, shape=slot_shape, dtype=var.dtype)
    else:
      initial_value = initializer

    if isinstance(var, kv_variable_ops.EmbeddingVariable):
      if slot_config is None:
        weight = get_embedding_variable_internal(
          name=f"{var._shared_name}/{slot_name}",
          initializer=initializer,
          trainable=False,
          embedding_dim=slot_shape,
          key_dtype=var._invalid_key_type,
          value_dtype=var.dtype,
          validate_shape=slot_shape.is_fully_defined(),
          steps_to_live=var._steps_to_live,
          ht_partition_num=var._ht_partition_num,
        )
        # _set_init_op_embedding_type_attr(weight, config_pb2.EmbeddingVariableType.MUTABLE)
      else:
        filter_strategy = None
        if var._filter_freq != 0:
          if var._max_element_size != 0:
            filter_strategy = ev_variables.CBFFilter(
              filter_freq=var._filter_freq,
              max_element_size=var._max_element_size,
              false_positive_probability=var._false_positive_probability,
              counter_type=var._counter_type,
            )
          else:
            filter_strategy = ev_variables.CounterFilter(filter_freq=var._filter_freq)
        if slot_config.slot_type is config_pb2.SlotType.EMBEDDING_VARIABLE:
          # _set_init_op_slot_num_attr(var, slot_config.slot_num)
          var._slot_num = slot_config.slot_num
          emb_index = var._emb_index
          if var.block_num > 1:
            var = var._primary
          weight = get_embedding_variable_v2_internal(
            name=f"{var._shared_name}/{slot_name}",
            initializer=initializer,
            trainable=False,
            embedding_dim=slot_shape,
            key_dtype=var._invalid_key_type,
            value_dtype=var.dtype,
            validate_shape=slot_shape.is_fully_defined(),
            evconfig=ev_variables.EmbeddingVariableConfig(
              steps_to_live=var._steps_to_live,
              handle_name=var._block_handle_name,
              emb_index=emb_index,
              block_num=var.block_num,
              slot_index=slot_config.slot_index,
              primary=var._primary,
              slot_num=slot_config.slot_num,
              storage_type=var.storage_type,
              storage_path=var._storage_path,
              storage_size=var._storage_size,
              storage_cache_strategy=var._storage_cache_strategy,
              layout=var._layout,
              l2_weight_threshold=var._l2_weight_threshold,
              filter_strategy=filter_strategy,
            ),
          )
        else:
          weight = tf.Variable(
            name=f"{var._shared_name}/{slot_name}",
            dtype=var.dtype,
            trainable=False,
            initial_value=initial_value,
          )
    else:
      with self._distribution_strategy_scope():
        strategy = tf.distribute.get_strategy()
        if not strategy.extended.variable_created_in_scope(var):
          raise ValueError(
            "Trying to create optimizer slot variable under the "
            "scope for tf.distribute.Strategy ({}), which is "
            "different from the scope used for the original "
            "variable ({}). Make sure the slot variables are "
            "created under the same strategy scope. This may "
            "happen if you're restoring from a checkpoint "
            "outside the scope.".format(strategy, var)
          )

        with strategy.extended.colocate_vars_with(var):
          weight = tf.Variable(
            name=f"{var._shared_name}/{slot_name}",
            dtype=var.dtype,
            trainable=False,
            initial_value=initial_value,
          )

    backend.track_variable(weight)
    slot_dict[slot_name] = weight
    self._restore_slot_variable(slot_name=slot_name, variable=var, slot_variable=weight)
    self._weights.append(weight)
  return weight


def _deduplicate_indexed_slices_with_counts(values, indices):
  """Sums `values` associated with any non-unique `indices`
  and return counts of each count in `values`."""
  unique_indices, new_index_positions, indices_counts = gen_array_ops.deepray_unique_with_counts(
    indices, out_idx=dtypes.int64
  )
  summed_values = math_ops.unsorted_segment_sum(values, new_index_positions, array_ops.shape(unique_indices)[0])
  return summed_values, unique_indices, indices_counts


def _deduplicate_indexed_slices_with_counts_reduction(values, indices, extra_counts, extra_indices):
  """Sums `values` associated with any non-unique `indices`
  and return counts of each count in `values`."""
  unique_indices, new_index_positions, summed_counts = gen_array_ops.deepray_unique_with_extra_counts(
    indices, extra_indices, extra_counts
  )
  summed_values = math_ops.unsorted_segment_sum(values, new_index_positions, array_ops.shape(unique_indices)[0])
  return summed_values, unique_indices, summed_counts


def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices, **kwargs):
  """Add ops to apply sparse gradients to `handle`, with repeated indices.

  Optimizers which override this method must deal with repeated indices. See
  the docstring of `_apply_sparse_duplicate_indices` for details. By default
  the correct behavior, to sum non-unique indices and their associated
  gradients, is enforced by first pre-processing `grad` and `indices` and
  passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
  with duplicate indices may instead override this method to avoid the
  overhead of summing.

  Args:
    grad: a `Tensor` representing the gradient for the affected indices.
    handle: a `Tensor` of dtype `resource` which points to the variable
     to be updated.
    indices: a `Tensor` of integral type representing the indices for
     which the gradient is nonzero. Indices may be repeated.

  Returns:
    An `Operation` which updates the value of the variable.
  """
  from deepray.custom_ops.embedding_variable import kv_variable_ops

  if isinstance(handle, kv_variable_ops.EmbeddingVariable) and handle.need_counts():
    if len(handle._counts_tensor.keys()) == 0:
      summed_grad, unique_indices, indices_counts = _deduplicate_indexed_slices_with_counts(
        values=grad, indices=indices
      )
    else:
      extra_counts, extra_indices = [], []
      if indices.op.type == "ConcatV2":
        for tensor in indices.op.inputs:
          if tensor.op.type == "Reshape":
            indices_tensor = tensor.op.inputs[0]
            if indices_tensor in handle._counts_tensor:
              extra_counts.append(handle._counts_tensor[indices_tensor])
              extra_indices.append(indices_tensor)
      elif indices.op.type == "Reshape":
        indices_tensor = indices.op.inputs[0]
        if indices_tensor in handle._counts_tensor:
          extra_counts.append(handle._counts_tensor[indices_tensor])
          extra_indices.append(indices_tensor)
      summed_grad, unique_indices, indices_counts = _deduplicate_indexed_slices_with_counts_reduction(
        grad, indices, extra_counts, extra_indices
      )
    return self._resource_apply_sparse(
      grad=summed_grad, var=handle, indices=unique_indices, indices_counts=indices_counts, **kwargs
    )
  else:
    summed_grad, unique_indices = _deduplicate_indexed_slices(values=grad, indices=indices)
    return self._resource_apply_sparse(summed_grad, handle, unique_indices, **kwargs)
