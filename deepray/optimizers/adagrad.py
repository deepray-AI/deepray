# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Adagrad for Deepray."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from absl import flags

from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
from deepray.custom_ops.embedding_variable import kv_variable_ops
from .ev_optimizer_patch import add_slot, SlotConfig, _resource_apply_sparse_duplicate_indices


class Adagrad(tf.keras.optimizers.legacy.Adagrad):
  def __init__(self, learning_rate=0.001, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    self.global_step = None
    flags.FLAGS([sys.argv[0], f"--ev_slot_num={1}"])

  def _create_slots(self, var_list):
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = tf.compat.v1.constant_initializer(self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, "accumulator", init, slot_config=SlotConfig(slot_index=1, slot_num=1))

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None, indices_counts=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)

    acc = self.get_slot(var, "accumulator")
    if isinstance(var, kv_variable_ops.EmbeddingVariable):
      if indices_counts != None:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adagrad_with_counts(
          var.handle,
          acc.handle,
          coefficients["lr_t"],
          grad,
          indices,
          self.global_step,
          indices_counts,
          use_locking=self._use_locking,
        )
      else:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adagrad(
          var.handle, acc.handle, coefficients["lr_t"], grad, indices, self.global_step, use_locking=self._use_locking
        )
    else:
      return tf.raw_ops.ResourceSparseApplyAdagradV2(
        var=var.handle,
        accum=acc.handle,
        lr=coefficients["lr_t"],
        epsilon=coefficients["epsilon"],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking,
      )


Adagrad.add_slot = add_slot
Adagrad._resource_apply_sparse_duplicate_indices = _resource_apply_sparse_duplicate_indices
