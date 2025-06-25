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
"""GradientDescentOptimizer for Deepray."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tf_keras.src.optimizers.legacy import gradient_descent as gd_old

from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
from deepray.custom_ops.embedding_variable import kv_variable_ops


class SGD(gd_old.SGD):
  def __init__(self, learning_rate=0.01, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    self.global_step = None

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = kwargs.get("apply_state", {}).get((var_device, var_dtype)) or self._fallback_apply_state(
      var_device, var_dtype
    )
    if self._momentum:
      # This method is only needed for momentum optimization.
      momentum_var = self.get_slot(var, "momentum")
      return tf.raw_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov,
      )
    else:
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        if var.need_counts() and len(var._counts_tensor.keys()) != 0:
          extra_counts, extra_indices = [], []
          if indices.op.type == "ConcatV2":
            for tensor in indices.op.inputs:
              if tensor.op.type == "Reshape":
                indices_tensor = tensor.op.inputs[0]
                if indices_tensor in var._counts_tensor:
                  extra_counts.append(var._counts_tensor[indices_tensor])
                  extra_indices.append(indices_tensor)
          elif indices.op.type == "Reshape":
            indices_tensor = indices.op.inputs[0]
            if indices_tensor in var._counts_tensor:
              extra_counts.append(var._counts_tensor[indices_tensor])
              extra_indices.append(indices_tensor)

          from deepray.custom_ops.unique_ops import gen_array_ops

          unique_indices, new_index_positions, indices_counts = gen_array_ops.deepray_unique_with_extra_counts(
            indices, extra_indices, extra_counts
          )
          summed_grads = math_ops.unsorted_segment_sum(grad, new_index_positions, array_ops.shape(unique_indices)[0])
          return gen_kv_variable_ops.kv_resource_sparse_apply_gradient_descent_with_counts(
            var.handle,
            coefficients["lr_t"],
            summed_grads,
            unique_indices,
            self.global_step,
            indices_counts,
            use_locking=self._use_locking,
          )
        else:
          return gen_kv_variable_ops.kv_resource_sparse_apply_gradient_descent(
            var.handle, coefficients["lr_t"], grad, indices, self.global_step, use_locking=self._use_locking
          )
      else:
        return tf.raw_ops.ResourceScatterAdd(
          resource=var.handle,
          indices=indices,
          updates=-grad * coefficients["lr_t"],
        )
