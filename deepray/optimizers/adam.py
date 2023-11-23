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
from tensorflow.python.keras.optimizer_v2 import adam as tf_adam

from deepray.custom_ops.training_ops import gen_training_ops


class Adam(tf_adam.Adam):
  """Deepray Adam optimizer for efficient sparse updates"""

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (
        (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
    )
    # beta1_power, beta2_power = self._get_beta_accumulators()
    return gen_training_ops.resource_sparse_apply_adam(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=coefficients['beta_1_power'],
        beta2_power=coefficients['beta_2_power'],
        lr=coefficients['lr_t'],
        beta1=coefficients['beta_1_t'],
        beta2=coefficients['beta_2_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking
    )
