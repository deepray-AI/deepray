# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Multiple Optimizer for TensorFlow.
References:
1. https://github.com/tensorflow/recommenders/blob/7caed557b9d5194202d8323f2d4795231a5d0b1d/tensorflow_recommenders/experimental/optimizers/composite_optimizer.py#L25
2. https://github.com/tensorflow/addons/blob/d208d752e98c310280938efa939117bf635a60a8/tensorflow_addons/optimizers/discriminative_layer_training.py#L47
3. https://github.com/NVIDIA-Merlin/models/blob/eb1e54196a64a70950b2a7e7744d2150e052d53e/merlin/models/tf/blocks/optimizer.py#L73
"""

from collections import defaultdict
from typing import List, Union

import tensorflow as tf
from packaging.version import Version
from typeguard import typechecked

from deepray.optimizers import KerasLegacyOptimizer

if Version(tf.__version__).release >= Version("2.16").release:
  # Determine if loading keras 2 or 3.
  if hasattr(tf.keras, "version") and Version(tf.keras.version()).release >= Version("3.0").release:
    # New versions of Keras require importing from `keras.src` when
    # importing internal symbols.
    from keras.src import backend
    from keras.src.utils import tf_utils
  else:
    from tf_keras.src import backend
    from tf_keras.src.utils import tf_utils
elif Version(tf.__version__).release >= Version("2.13").release:
  from keras.src import backend
  from keras.src.utils import tf_utils
else:
  from keras import backend
  from keras.utils import tf_utils


class MultiOptimizer(KerasLegacyOptimizer):
  """Multi Optimizer Wrapper for Discriminative Layer Training.

  Creates a wrapper around a set of instantiated optimizer layer pairs.
  Generally useful for transfer learning of deep networks.

  Each optimizer will optimize only the weights associated with its paired layer.
  This can be used to implement discriminative layer training by assigning
  different learning rates to each optimizer layer pair.
  `(tf.keras.optimizers.legacy.Optimizer, List[tf.keras.layers.Layer])` pairs are also supported.
  Please note that the layers must be instantiated before instantiating the optimizer.

  Args:
      optimizers_and_varnames: a list of tuples of an optimizer and a layer or model.
          Each tuple should contain exactly 1 instantiated optimizer and 1 object that
          subclasses `tf.keras.Model`, `tf.keras.Sequential` or `tf.keras.layers.Layer`.
          Nested layers and models will be automatically discovered.
          Alternatively, in place of a single layer, you can pass a list of layers.
      default_optimizer: Default optimizer for the left trainable variables.

  Usage:

  >>> model = tf.keras.Sequential([
  ...     tf.keras.Input(shape=(4,)),
  ...     tf.keras.layers.Dense(8, name="varname1"),
  ...     tf.keras.layers.Dense(16, name="varname2"),
  ...     tf.keras.layers.Dense(32, name="varname3"),
  ...     tf.keras.layers.Dense(64),
  ... ])
  >>> optimizers = [
  ...     tf.keras.optimizers.Adam(learning_rate=1e-4),
  ...     tf.keras.optimizers.Adam(learning_rate=1e-2),
  ...     tf.keras.optimizers.Adam(learning_rate=1e-3)
  ... ]
  >>> optimizers_and_varnames = [(optimizers[0], "varname1"), (optimizers[1], "varname2,varname3")]
  >>> optimizer = dp.optimizers.MultiOptimizer(optimizers_and_varnames, optimizers[2])
  >>> model.compile(optimizer=optimizer, loss="mse")

  Reference:
      - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
      - [Collaborative Layer-wise Discriminative Learning in Deep Neural Networks](https://arxiv.org/abs/1607.05440)

  Note: Currently, `dp.optimizers.MultiOptimizer` does not support callbacks that modify optimizers.
      However, you can instantiate optimizer layer pairs with
      `tf.keras.optimizers.schedules.LearningRateSchedule`
      instead of a static learning rate.

  This code should function on CPU, GPU, and TPU. Apply with `tf.distribute.Strategy().scope()` context as you
  would with any other optimizer.
  """

  @typechecked
  def __init__(
    self,
    optimizers_and_varnames: Union[list, None] = None,
    default_optimizer: KerasLegacyOptimizer = None,
    name: str = "MultiOptimizer",
    **kwargs,
  ):
    super(MultiOptimizer, self).__init__(name, **kwargs)
    if default_optimizer is None:
      raise RuntimeError("Must specify a `default_optimizer`.")
    self.optimizers_and_varnames = optimizers_and_varnames
    self.default_optimizer = default_optimizer

  def apply_gradients(self, grads_and_vars, **kwargs):
    """Wrapped apply_gradient method.

    Returns an operation to be executed.
    """
    # Create a dictionary with a default optimizer and an empty variable list
    grad_var_dict = defaultdict(list)

    # Iterate over the trainable variables list
    for grad, var in grads_and_vars:
      # Check if each variable name exists in the variable name list
      for optimizer, varnames in self.optimizers_and_varnames:
        if any(name in var.name for name in varnames.split(",")):
          # If it does, append the variable to the optimizer's variable list
          grad_var_dict[optimizer].append((grad, var))
          break
      else:
        # If it doesn't, append the variable to the default optimizer's variable list
        grad_var_dict[self.default_optimizer].append((grad, var))

    update_ops = []
    # Call the apply_gradients method for each optimizer with the corresponding gradient and variable list
    for optimizer, grad_var in grad_var_dict.items():
      update_ops.append(optimizer.apply_gradients(grad_var, **kwargs))

    # update_ops = [optimizer.apply_gradients(grad_var, **kwargs) for optimizer, grad_var in grad_var_dict.items()]
    update_group = tf.group(update_ops)

    any_symbolic = any(isinstance(i, tf.Operation) or tf_utils.is_symbolic_tensor(i) for i in update_ops)

    if not tf.executing_eagerly() or any_symbolic:
      # If the current context is graph mode or any of the update ops are
      # symbolic then the step update should be carried out under a graph
      # context. (eager updates execute immediately)
      with backend._current_graph(  # pylint: disable=protected-access
        update_ops
      ).as_default():
        with tf.control_dependencies([update_group]):
          return self.iterations.assign_add(1, read_value=False)

    return self.iterations.assign_add(1)

  def variables(self):
    """Returns the optimizer's variables."""
    # OptimizerV2.variables() returns self._weights, so override that method.
    return self.weights

  @property
  def weights(self) -> List[tf.Variable]:
    """Returns the optimizer's variables."""
    weights = []
    for optimizer, _ in self.optimizers_and_varnames:
      weights += optimizer.weights
    return weights

  @property
  def optimizers(self) -> List[tf.keras.optimizers.legacy.Optimizer]:
    """Returns the optimizers in composite optimizer (in the original order)."""
    return [optimizer for optimizer, _ in self.optimizers_and_varnames]
