# Copyright 2023 The TFPlus Authors. All rights reserved.
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
"""Adam optimizer for Deepray. We inherit TensorFlow AdamOptimizer to
implement a variant of AdamOptimizer, which can handle sparse updates
more efficiently than the original one. This variant is written based
on the idea proposed in LazyAdamOptimizer. Please refer to
https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/opt/python/training/lazy_adam_optimizer.py.
"""

from __future__ import absolute_import, division, print_function

import sys

from absl import flags
from tf_keras.src.optimizers.legacy import adam as adam_old

from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
from deepray.custom_ops.embedding_variable import kv_variable_ops
from deepray.custom_ops.training_ops import gen_training_ops
from .ev_optimizer_patch import add_slot, SlotConfig, _resource_apply_sparse_duplicate_indices


class Adam(adam_old.Adam):
  """Deepray Adam optimizer for efficient sparse updates"""

  def __init__(self, learning_rate=0.001, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    self.global_step = None
    flags.FLAGS([sys.argv[0], f"--ev_slot_num={2}"])

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, "m", slot_config=SlotConfig(slot_index=1, slot_num=2))
    for var in var_list:
      self.add_slot(var, "v", slot_config=SlotConfig(slot_index=2, slot_num=2))
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, "vhat")

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None, indices_counts=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    if isinstance(var, kv_variable_ops.EmbeddingVariable):
      if indices_counts is not None:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adam_with_counts(
          var.handle,
          m.handle,
          v.handle,
          coefficients["beta_1_power"],
          coefficients["beta_2_power"],
          coefficients["lr_t"],
          coefficients["beta_1_t"],
          coefficients["beta_2_t"],
          coefficients["epsilon"],
          grad,
          indices,
          self.global_step,
          indices_counts,
          use_locking=self._use_locking,
        )
      else:
        return gen_kv_variable_ops.kv_resource_sparse_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          coefficients["beta_1_power"],
          coefficients["beta_2_power"],
          coefficients["lr_t"],
          coefficients["beta_1_t"],
          coefficients["beta_2_t"],
          coefficients["epsilon"],
          grad,
          indices,
          self.global_step,
          use_locking=self._use_locking,
        )
    else:
      return gen_training_ops.resource_sparse_apply_adam(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=coefficients["beta_1_power"],
        beta2_power=coefficients["beta_2_power"],
        lr=coefficients["lr_t"],
        beta1=coefficients["beta_1_t"],
        beta2=coefficients["beta_2_t"],
        epsilon=coefficients["epsilon"],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking,
      )


Adam.add_slot = add_slot
Adam._resource_apply_sparse_duplicate_indices = _resource_apply_sparse_duplicate_indices
