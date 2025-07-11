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
"""Multiple Optimizer for TensorFlow.
References:
1. https://github.com/tensorflow/recommenders/blob/7caed557b9d5194202d8323f2d4795231a5d0b1d/tensorflow_recommenders/experimental/optimizers/composite_optimizer.py#L25
2. https://github.com/tensorflow/addons/blob/d208d752e98c310280938efa939117bf635a60a8/tensorflow_addons/optimizers/discriminative_layer_training.py#L47
3. https://github.com/NVIDIA-Merlin/models/blob/eb1e54196a64a70950b2a7e7744d2150e052d53e/merlin/models/tf/blocks/optimizer.py#L73
"""

import collections
from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tf_keras
from typeguard import typechecked

from deepray.optimizers import KerasLegacyOptimizer

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


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
    self.built = False
    self.var_optimizer_dict = {}

  def build(self, grads_and_vars):
    for grad, var in grads_and_vars:
      # Check if each variable name exists in the variable name list
      for optimizer, varnames in self.optimizers_and_varnames:
        if any(name in var.name for name in varnames.split(",")):
          # If it does, append the variable to the optimizer's variable list
          self.var_optimizer_dict[var.ref()] = optimizer
          break
      else:
        # If it doesn't, append the variable to the default optimizer's variable list
        self.var_optimizer_dict[var.ref()] = self.default_optimizer
    self.built = True

  def apply_gradients(
    self,
    grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
    name: Optional[str] = None,
    experimental_aggregate_gradients: bool = True,
  ) -> None:
    """Wrapped apply_gradient method.

    Returns an operation to be executed.
    """
    if not self.built:
      self.build(grads_and_vars)
    # Create a dictionary with a default optimizer and an empty variable list
    optimizer_grads_and_vars = collections.defaultdict(list)
    # Iterate over the trainable variables list
    for grad, var in grads_and_vars:
      if var.ref() in self.var_optimizer_dict:
        optimizer = self.var_optimizer_dict[var.ref()]
        optimizer_grads_and_vars[optimizer].append((grad, var))
        # for variables not in optimizers_and_blocks, assign default optimizer
      else:
        optimizer_grads_and_vars[self.default_optimizer].append((grad, var))

    # Apply gradient for each optimizer
    for optimizer, opt_grads_and_vars in optimizer_grads_and_vars.items():
      optimizer.apply_gradients(
        opt_grads_and_vars,
        name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients,
      )

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
  def optimizers(self) -> List[tf_keras.optimizers.legacy.Optimizer]:
    """Returns the optimizers in composite optimizer (in the original order)."""
    return [optimizer for optimizer, _ in self.optimizers_and_varnames]
