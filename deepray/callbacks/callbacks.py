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
"""Callbacks: utilities called at certain points during model training."""

import numpy as np
import tensorflow as tf
from absl import flags
from tf_keras import callbacks as callbacks_module


def sync_to_numpy_or_python_type(tensors):
  """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python
  scalar types.

  For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
  it converts it to a Python type, such as a float or int, by calling
  `result.item()`.

  Numpy scalars are converted, as Python types are often more convenient to
  deal with. This is especially useful for bfloat16 Numpy scalars, which don't
  support as many operations as other Numpy values.

  Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
  forced to
  sync during this process.

  Args:
    tensors: A structure of tensors.

  Returns:
    `tensors`, but scalar tensors are converted to Python types and non-scalar
    tensors are converted to Numpy arrays.
  """
  if isinstance(tensors, tf.distribute.experimental.coordinator.RemoteValue):
    tensors = tensors.fetch()
  if isinstance(tensors, list) and isinstance(tensors[0], tf.distribute.experimental.coordinator.RemoteValue):
    tensors = tf.nest.map_structure(lambda t: t.fetch(), tensors)

  def _to_single_numpy_or_python_type(t):
    if flags.FLAGS.use_horovod:
      import horovod.tensorflow.keras as hvd

      t = hvd.allreduce(t, op=hvd.Average)
    # Don't turn ragged or sparse tensors to NumPy.
    if isinstance(t, tf.Tensor):
      t = t.numpy()
    # Strings, ragged and sparse tensors don't have .item(). Return them
    # as-is.
    if not isinstance(t, (np.ndarray, np.generic)):
      return t
    return t.item() if np.ndim(t) == 0 else t

  return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)


class HvdCallbackList(callbacks_module.CallbackList):
  def _process_logs(self, logs, is_batch_hook=False):
    """Turns tensors into numpy arrays or Python scalars if necessary."""
    if logs is None:
      return {}
    if self._supports_tf_logs:
      return logs
    if is_batch_hook and self._batch_hooks_support_tf_logs:
      return logs
    return sync_to_numpy_or_python_type(logs)
