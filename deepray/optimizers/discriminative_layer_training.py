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
"""Discriminative Layer Training Optimizer for TensorFlow."""

from collections import defaultdict
from typing import Union

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer as keras_optimizer
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.training import optimizer
from typeguard import typechecked

from deepray.optimizers import KerasLegacyOptimizer


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
    self.optimizers_and_varnames = optimizers_and_varnames
    self.default_optimizer = default_optimizer
    if optimizers_and_varnames is None and default_optimizer is None:
      raise RuntimeError(
        "Must specify both of `optimizers_and_varnames` and `default_optimizer`."
      )

    if isinstance(self, optimizer.Optimizer):
      self.compute_gradients = self.default_optimizer.compute_gradients
    elif isinstance(self, optimizer_v2.OptimizerV2) or isinstance(
        self, keras_optimizer):
      self.compute_gradients = self.default_optimizer._compute_gradients
    else:
      raise Exception("Optimizer type is not supported! got {}".format(
        str(type(self))))

  def minimize(self, loss, var_list, tape):
    # Compute gradients
    grads_and_vars = self.compute_gradients(loss=loss, var_list=var_list, tape=tape)
    self.apply_gradients(grads_and_vars)

  def apply_gradients(self, grads_and_vars, name=None, **kwargs):
    # Create a dictionary with a default optimizer and an empty variable list
    var_dict, grad_dict = defaultdict(list), defaultdict(list)

    # Iterate over the trainable variables list
    for grad, var in grads_and_vars:
      # Check if each variable name exists in the variable name list
      for optimizer, varnames in self.optimizers_and_varnames:
        if any(name in var.name for name in varnames.split(',')):
          # If it does, append the variable to the optimizer's variable list
          var_dict[optimizer].append(var)
          grad_dict[optimizer].append(grad)
          break
      else:
        # If it doesn't, append the variable to the default optimizer's variable list
        var_dict[self.default_optimizer].append(var)
        grad_dict[self.default_optimizer].append(grad)

    # Call the apply_gradients method for each optimizer with the corresponding gradient and variable list
    for optimizer, partvar_list in var_dict.items():
      optimizer.apply_gradients(zip(grad_dict[optimizer], partvar_list))

  @property
  def iterations(self):
    """The number of training steps this `optimizer` has run.

    By default, iterations would be incremented by one every time
    `apply_gradients()` is called.
    """
    return self.default_optimizer.iterations
