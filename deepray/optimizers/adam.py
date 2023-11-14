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

from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam as tf_adam

from deepray.custom_ops.training_ops import gen_training_ops


class AdamOptimizer(tf_adam.AdamOptimizer):
  """Deepray Adam optimizer for efficient sparse updates"""

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return gen_training_ops.sparse_apply_adam(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking
    )

  def _resource_apply_sparse_shared(self, grad, var, indices, scatter_add):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return gen_training_ops.resource_sparse_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking
    )
