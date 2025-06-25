# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from typing import List, Optional, Union

import tensorflow as tf
from absl import flags

from .warmup import WarmUpPolynomial


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, optimizer_type="adam"):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  if optimizer_type == "adam":
    power = 1.0
    decayed_learning_rate_at_crossover_point = init_lr * (
      (1.0 - float(num_warmup_steps) / float(num_train_steps)) ** power
    )
  else:
    power = 0.5
    decayed_learning_rate_at_crossover_point = init_lr
  init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
  print(
    "decayed_learning_rate_at_crossover_point = %e, adjusted_init_lr = %e"
    % (decayed_learning_rate_at_crossover_point, init_lr)
  )

  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=0.0, power=power
  )
  if num_warmup_steps:
    learning_rate_fn = WarmUpPolynomial(
      initial_learning_rate=init_lr, decay_schedule_fn=learning_rate_fn, warmup_steps=num_warmup_steps
    )
  if optimizer_type == "adamw":
    optimizer = AdamWeightDecay(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )
  elif optimizer_type == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
  else:
    skip_list = ["None"]  # to avoid exclude_from_layer_adaptation set to exclude_from_weight_decay if the arg is None
    import deepray.optimizers as dp_optimizers

    optimizer = dp_optimizers.LAMB(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
      exclude_from_layer_adaptation=skip_list,
    )
  # Horovod: add Horovod DistributedOptimizer.
  # ValueError: Unknown decay: WarmUp. Please ensure this object is passed to the `custom_objects` argument. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.
  # if FLAGS.use_horovod:
  #   import horovod.tensorflow.keras as hvd
  #   optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)
  return optimizer


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """
  Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
  loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
  with the m and v parameters in strange ways as shown in [Decoupled Weight Decay
  Regularization](https://arxiv.org/abs/1711.05101).

  Instead we want to decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
  to adding the square of the weights to the loss with plain (non-momentum) SGD.

  Args:
      learning_rate (`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, *optional*, defaults to 1e-3):
          The learning rate to use or a schedule.
      beta_1 (`float`, *optional*, defaults to 0.9):
          The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
      beta_2 (`float`, *optional*, defaults to 0.999):
          The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
      epsilon (`float`, *optional*, defaults to 1e-7):
          The epsilon parameter in Adam, which is a small constant for numerical stability.
      amsgrad (`bool`, *optional*, default to `False`):
          Whether to apply AMSGrad variant of this algorithm or not, see [On the Convergence of Adam and
          Beyond](https://arxiv.org/abs/1904.09237).
      weight_decay_rate (`float`, *optional*, defaults to 0):
          The weight decay to apply.
      include_in_weight_decay (`List[str]`, *optional*):
          List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
          applied to all parameters by default (unless they are in `exclude_from_weight_decay`).
      exclude_from_weight_decay (`List[str]`, *optional*):
          List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
          `include_in_weight_decay` is passed, the names in it will supersede this list.
      name (`str`, *optional*, defaults to 'AdamWeightDecay'):
          Optional name for the operations created when applying gradients.
      kwargs:
          Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
          norm; `clipvalue` is clip gradients by value, `decay` is included for backward compatibility to allow time
          inverse decay of learning rate. `lr` is included for backward compatibility, recommended to use
          `learning_rate` instead.
  """

  def __init__(
    self,
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
    amsgrad: bool = False,
    weight_decay_rate: float = 0.0,
    include_in_weight_decay: Optional[List[str]] = None,
    exclude_from_weight_decay: Optional[List[str]] = None,
    name: str = "AdamWeightDecay",
    **kwargs,
  ):
    super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config with WarmUp custom object."""
    custom_objects = {"WarmUp": WarmUpPolynomial}
    return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
      self.weight_decay_rate, name="adam_weight_decay_rate"
    )

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
        learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
        use_locking=self._use_locking,
      )
    return tf.no_op()

  def apply_gradients(self, grads_and_vars, name=None, **kwargs):
    grads, tvars = list(zip(*grads_and_vars))
    return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients["lr_t"], {"apply_state": apply_state}

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

  def get_config(self):
    config = super().get_config()
    config.update({"weight_decay_rate": self.weight_decay_rate})
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False

    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator:
  """Gradient accumulation utility.

  When used with a distribution strategy, the accumulator should be called in a
  replica context. Gradients will be accumulated locally on each replica and
  without synchronization. Users should then call ``.gradients``, scale the
  gradients if required, and pass the result to ``apply_gradients``.
  """

  def __init__(self):
    """Initializes the accumulator."""
    self._gradients = []
    self._accum_steps = None

  @property
  def step(self):
    """Number of accumulated steps."""
    if self._accum_steps is None:
      self._accum_steps = tf.Variable(
        tf.constant(0, dtype=tf.int64),
        trainable=False,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
      )
    return self._accum_steps.value()

  @property
  def gradients(self):
    if not self._gradients:
      raise ValueError("The accumulator should be called first to initialize the gradients")
    return list(gradient.value() if gradient is not None else None for gradient in self._gradients)

  def reset(self):
    if not self._gradients:
      return
    self._accum_steps.assign(0)
    for gradient in self._gradients:
      if gradient is not None:
        gradient.assign(tf.zeros(tf.shape(gradient), dtype=gradient.dtype))

  def add_gradients(self, grads):
    if not self._gradients:
      _ = self.step
      self._gradients.extend([
        tf.Variable(tf.zeros_like(g), trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
        if g is not None
        else None
        for g in grads
      ])
    if len(grads) != len(self._gradients):
      raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(grads)))

    for accum_grad, grad in zip(self._gradients, grads):
      if accum_grad is not None:
        accum_grad.assign_add(grad)

    self._accum_steps.assign_add(1)
