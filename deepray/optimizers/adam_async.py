# Copyright 2025 The Deepray Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AdamAsync optimizer for Deepray."""

from __future__ import absolute_import, division, print_function

import sys

import tensorflow as tf
from absl import flags
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from deepray.custom_ops.embedding_variable import config_pb2
from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
from deepray.custom_ops.embedding_variable import kv_variable_ops
from deepray.custom_ops.training_ops import gen_training_ops
from .ev_optimizer_patch import add_slot, SlotConfig, _resource_apply_sparse_duplicate_indices


class AdamAsync(tf.keras.optimizers.legacy.Adam):
  """Deepray Adam optimizer for efficient sparse updates"""

  def __init__(self, learning_rate=0.001, apply_sparse_rmsprop=False, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    self._apply_sparse_rmsprop = apply_sparse_rmsprop
    self.global_step = None
    flags.FLAGS([sys.argv[0], f"--ev_slot_num={2}"])

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, "m", slot_config=SlotConfig(slot_index=1, slot_num=2))
      # for var in var_list:
      self.add_slot(var, "v", slot_config=SlotConfig(slot_index=2, slot_num=2))
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        self.add_slot(
          var,
          slot_name="beta1_power",
          initializer=array_ops.expand_dims(self._get_hyper("beta_1", var.dtype.base_dtype), -1),
          slot_config=SlotConfig(slot_type=config_pb2.SlotType.VARIABLE),
        )
        self.add_slot(
          var,
          slot_name="beta2_power",
          initializer=array_ops.expand_dims(self._get_hyper("beta_2", var.dtype.base_dtype), -1),
          slot_config=SlotConfig(slot_type=config_pb2.SlotType.VARIABLE),
        )
      else:
        self.add_slot(
          var,
          slot_name="beta1_power",
          initializer=self._get_hyper("beta_1", var.dtype.base_dtype),
          slot_config=SlotConfig(slot_type=config_pb2.SlotType.VARIABLE),
        )
        self.add_slot(
          var,
          slot_name="beta2_power",
          initializer=self._get_hyper("beta_2", var.dtype.base_dtype),
          slot_config=SlotConfig(slot_type=config_pb2.SlotType.VARIABLE),
        )
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, "vhat")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    if "learning_rate" in self._hyper:
      lr_t = tf.identity(self._decayed_lr(var_dtype))
      apply_state[(var_device, var_dtype)]["lr_t"] = lr_t

    beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
    beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
    # beta_1_power = tf.identity(self._get_hyper("beta1_power", var_dtype))
    # beta_2_power = tf.identity(self._get_hyper("beta2_power", var_dtype))

    # lr = apply_state[(var_device, var_dtype)]["lr_t"] * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power))
    apply_state[(var_device, var_dtype)].update(
      dict(
        # lr=lr,
        epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        # beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
        beta_2_t=beta_2_t,
        # beta_2_power=beta_2_power,
        one_minus_beta_2_t=1 - beta_2_t,
      )
    )

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power = self.get_slot(var, "beta1_power")
    beta2_power = self.get_slot(var, "beta2_power")
    return gen_training_ops.resource_apply_adam_async(
      var.handle,
      m.handle,
      v.handle,
      beta1_power.handle,
      beta2_power.handle,
      math_ops.cast(self._lr_t, grad.dtype.base_dtype),
      math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
      math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
      math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
      grad,
      use_locking=self._use_locking,
      apply_sparse_rmsprop=self._apply_sparse_rmsprop,
    )

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None, indices_counts=None):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power = self.get_slot(var, "beta1_power")
    beta2_power = self.get_slot(var, "beta2_power")
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)

    if isinstance(var, kv_variable_ops.EmbeddingVariable):
      if indices_counts is not None:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adam_async_with_counts(
          var.handle,
          m.handle,
          v.handle,
          beta1_power.handle,
          beta2_power.handle,
          coefficients["lr_t"],
          coefficients["beta_1_t"],
          coefficients["beta_2_t"],
          coefficients["epsilon"],
          grad,
          indices,
          self.global_step,
          indices_counts,
          use_locking=self._use_locking,
          apply_sparse_rmsprop=self._apply_sparse_rmsprop,
        )
      else:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adam_async(
          var.handle,
          m.handle,
          v.handle,
          beta1_power.handle,
          beta2_power.handle,
          coefficients["lr_t"],
          coefficients["beta_1_t"],
          coefficients["beta_2_t"],
          coefficients["epsilon"],
          grad,
          indices,
          self.global_step,
          use_locking=self._use_locking,
          apply_sparse_rmsprop=self._apply_sparse_rmsprop,
        )
    else:
      return gen_training_ops.resource_sparse_apply_adam_async(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=beta1_power.handle,
        beta2_power=beta2_power.handle,
        lr=coefficients["lr_t"],
        beta1=coefficients["beta_1_t"],
        beta2=coefficients["beta_2_t"],
        epsilon=coefficients["epsilon"],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking,
        apply_sparse_rmsprop=self._apply_sparse_rmsprop,
      )


AdamAsync.add_slot = add_slot
AdamAsync._resource_apply_sparse_duplicate_indices = _resource_apply_sparse_duplicate_indices
