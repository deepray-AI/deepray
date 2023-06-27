# -*- coding:utf-8 -*-
"""
Author:
    DengGuo, dengguo@kanzhun.com
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from deepray.activations.dice import Dice
from tensorflow.python.keras import backend_config

epsilon = backend_config.epsilon

try:
  unicode
except NameError:
  unicode = str


def activation_layer(activation, name=None):
  if activation in ("dice", "Dice"):
    act_layer = Dice(name=name)
  elif isinstance(activation, (str, unicode)):
    act_layer = tf.keras.layers.Activation(activation, name=name)
  elif issubclass(activation, Layer):
    act_layer = activation(name=name)
  else:
    raise ValueError(
      "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
  return act_layer


class CGC3(Layer):
  """ Input shape
      - 2D tensor with shape: ``(batch_size, units)`` or ``(batch_size, t + 1, units)``
      - Or list of ``[input, gate_input]``, gate_input with shape ``(batch_size, 1, gate_input_dim)``
    Output shape
      - 2D tensor with shape: ``(batch_size, t, units)`` or ``(batch_size, t + 1, units)``.
    Arguments
      - **units** : expert tower, feed forward network layer
      - **num_experts_task** : Positive integer, number of experts for each task.
      - **num_experts_share** : Positive integer, number of experts for share.
      - **num_tasks** : Positive integer, number of tasks.
      - **output_share** : Boolean, whether to output share expert.
      - **use_bias** : Boolean, whether to use weight bias.
      - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
      - **seed**: A Python integer to use as random seed.
    References
      - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
  """

  def __init__(self, num_tasks=2,
               num_experts_task=1,
               num_experts_share=2,
               units=[256],
               activation="relu",
               output_activation=None,
               output_share=False,
               use_bias=True,
               l2_reg=0, seed=1024, **kwargs):
    self.units = units
    self.num_experts_task = num_experts_task
    self.num_experts_share = num_experts_share
    self.num_tasks = num_tasks
    self.activation = activation
    self.output_activation = output_activation
    self.output_share = output_share
    self.use_bias = use_bias
    self.l2_reg = l2_reg
    self.seed = seed
    super(CGC3, self).__init__(**kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, (list, tuple)):
      if len(input_shape[0]) == 3 and input_shape[0][1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %s, expect to be (batch_size, 1, dim) or (batch_size, %d, dim)" % (input_shape, self.num_tasks + 1))
      units = [input_shape[0][-1]] + self.units
    else:
      if len(input_shape) == 3 and input_shape[1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %s, expect to be (batch_size, 1, dim) or (batch_size, %d, dim)" % (input_shape, self.num_tasks + 1))
      units = [input_shape[-1]] + self.units

    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share
    # expert
    self.experts = []
    self.experts_bias = []
    for n in range(num_experts):
      self.experts.append([self.add_weight(name=f'experts_weight_{n}_{i}',
                                           shape=(units[i], units[i + 1]),
                                           initializer=tf.initializers.glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg),
                                           trainable=False) for i in range(len(units) - 1)])
      self.experts_bias.append([self.add_weight(name=f'experts_bias_{n}_{i}',
                                                shape=(self.units[i],),
                                                initializer=tf.initializers.Zeros(),
                                                trainable=False) for i in range(len(self.units))])

    self.gating = [tf.keras.layers.Dense(self.num_experts_task + self.num_experts_share,
                                         name="gating_" + str(i),
                                         use_bias=False,
                                         trainable=False) for i in range(self.num_tasks)]

    if self.output_share:
      self.gating_share = tf.keras.layers.Dense(num_experts, name="gating_share", use_bias=False, trainable=False)

    self.activation_layers = [activation_layer(self.output_activation if i == len(self.units) - 1 and self.output_activation else self.activation)
                              for i in range(len(self.units))]

    # Be sure to call this somewhere!
    super(CGC3, self).build(input_shape)

  def call(self, inputs):
    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share

    x1 = None
    if isinstance(inputs, (list, tuple)):
      target_inputs, gate_input = inputs
      x1 = [gate_input] * (self.num_tasks + 1)
    else:
      target_inputs = inputs

    if K.ndim(target_inputs) == 2:
      x0 = tf.tile(tf.expand_dims(target_inputs, 1), [1, num_experts, 1])
      if not x1:
        x1 = [tf.expand_dims(target_inputs, axis=1)] * (self.num_tasks + 1)
    elif K.ndim(target_inputs) == 3:
      x0 = tf.repeat(target_inputs, [self.num_experts_task] * self.num_tasks + [self.num_experts_share], axis=1)
      if not x1:
        x1 = tf.split(target_inputs, num_or_size_splits=(self.num_tasks + 1), axis=1)
    else:
      raise ValueError("Unexpected inputs dims %d" % K.ndim(target_inputs))

    x0 = tf.split(x0, num_or_size_splits=num_experts, axis=1)  # n * (bs, 1, d)
    for n in range(num_experts):
      x0[n] = tf.squeeze(x0[n], axis=1)
      for i in range(len(self.units)):
        # for gpu which not support einsum
        x0[n] = tf.matmul(x0[n], self.experts[n][i])  # (bs, e)
        x0[n] = self.activation_layers[i](tf.add(x0[n], self.experts_bias[n][i]))

    ret = []
    for i in range(self.num_tasks):
      gating_score = tf.nn.softmax(self.gating[i](x1[i]))  # (bs, 1, num_experts_task + num_experts_share)
      output_of_experts = tf.stack(x0[i * self.num_experts_task:(i + 1) * self.num_experts_task] + x0[-self.num_experts_share:], axis=1)  # (bs, num_experts_task + num_experts_share, unit)
      ret.append(tf.matmul(gating_score, output_of_experts))  # (bs, 1, unit)

    if self.output_share:
      gating_score = tf.nn.softmax(self.gating_share(x1[-1]))  # (bs, 1, num_experts)
      ret.append(tf.matmul(gating_score, tf.stack(x0, axis=1)))  # (bs, 1, unit)

    return tf.concat(ret, axis=1)

  def get_config(self, ):
    config = {'units': self.units,
              'num_experts_task': self.num_experts_task,
              'num_experts_share': self.num_experts_share,
              'num_tasks': self.num_tasks,
              'activation': self.activation,
              'output_activation': self.output_activation,
              'output_share': self.output_share,
              'use_bias': self.use_bias,
              'l2_reg': self.l2_reg,
              'seed': self.seed}
    base_config = super(CGC3, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    if self.output_share:
      return None, self.num_tasks + 1, self.units[-1]
    return None, self.num_tasks, self.units[-1]


class CGC2(Layer):
  """
        CPU Version, using tf.einsum op
    """

  def __init__(self, num_tasks=2,
               num_experts_task=1,
               num_experts_share=2,
               units=[256],
               activation="relu",
               output_activation=None,
               output_share=False,
               use_bias=True,
               l2_reg=0, seed=1024, **kwargs):
    self.units = units
    self.num_experts_task = num_experts_task
    self.num_experts_share = num_experts_share
    self.num_tasks = num_tasks
    self.activation = activation
    self.output_activation = output_activation
    self.output_share = output_share
    self.use_bias = use_bias
    self.l2_reg = l2_reg
    self.seed = seed
    super(CGC2, self).__init__(**kwargs)

  def build(self, input_shape):
    # TensorShape or [TensorShape, TensorShape]
    if isinstance(input_shape, (list, tuple)):
      if len(input_shape[0]) == 3 and input_shape[0][1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %d, expect to be (batch_size, 1, dim) or (batch_size, num_tasks + 1, dim)" % (input_shape,))
      units = [input_shape[0][-1]] + self.units
    else:
      if len(input_shape) == 3 and input_shape[1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %d, expect to be (batch_size, 1, dim) or (batch_size, num_tasks + 1, dim)" % (input_shape,))
      units = [input_shape[-1]] + self.units

    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share
    # expert
    self.experts = [self.add_weight(name='experts_weight_' + str(i),
                                    shape=(num_experts, units[i], units[i + 1]),
                                    initializer=tf.initializers.glorot_normal(seed=self.seed),
                                    regularizer=l2(self.l2_reg),
                                    trainable=True) for i in range(len(units) - 1)]
    self.experts_bias = [self.add_weight(name='experts_bias_' + str(i),
                                         shape=(num_experts, self.units[i]),
                                         initializer=tf.initializers.Zeros(),
                                         trainable=True) for i in range(len(self.units))]

    self.gating = [tf.keras.layers.Dense(self.num_experts_task + self.num_experts_share,
                                         name="gating_" + str(i),
                                         use_bias=False) for i in range(self.num_tasks)]

    if self.output_share:
      self.gating_share = tf.keras.layers.Dense(num_experts, name="gating_share", use_bias=False)

    self.activation_layers = [activation_layer(self.output_activation if i == len(self.units) - 1 and self.output_activation else self.activation)
                              for i in range(len(self.units))]

    # Be sure to call this somewhere!
    super(CGC2, self).build(input_shape)

  def call(self, inputs):
    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share
    # inputs -> x0 (batch_size, num_experts, dim)

    x1 = None
    if isinstance(inputs, (list, tuple)):
      target_inputs, gate_input = inputs
      x1 = [gate_input] * (self.num_tasks + 1)
    else:
      target_inputs = inputs

    if K.ndim(target_inputs) == 2:
      x0 = tf.tile(tf.expand_dims(target_inputs, 1), [1, num_experts, 1])
      if not x1:
        x1 = [tf.expand_dims(target_inputs, axis=1)] * (self.num_tasks + 1)
    elif K.ndim(target_inputs) == 3:
      x0 = tf.repeat(target_inputs, [self.num_experts_task] * self.num_tasks + [self.num_experts_share], axis=1)
      if not x1:
        x1 = tf.split(target_inputs, num_or_size_splits=(self.num_tasks + 1), axis=1)
    else:
      raise ValueError("Unexpected inputs dims %d" % K.ndim(target_inputs))

    for i in range(len(self.units)):
      # for cpu
      x0 = tf.einsum('bnd,nde->bne', x0, self.experts[i])  # (bs, num_experts, unit)
      # bias & activation
      x0 = tf.add(x0, self.experts_bias[i])
      x0 = self.activation_layers[i](x0)

    ret = []
    x2 = tf.split(x0, num_or_size_splits=[self.num_experts_task] * self.num_tasks + [self.num_experts_share], axis=1)
    for i in range(self.num_tasks):
      gating_score = tf.nn.softmax(self.gating[i](x1[i]))  # (bs, 1, num_experts_task + num_experts_share)
      output_of_experts = tf.concat([x2[i], x2[-1]], axis=1)  # (bs, num_experts_task + num_experts_share, unit)
      ret.append(tf.matmul(gating_score, output_of_experts))  # (bs, 1, unit)

    if self.output_share:
      gating_score = tf.nn.softmax(self.gating_share(x1[-1]))  # (bs, 1, num_experts)
      ret.append(tf.matmul(gating_score, x0))  # (bs, 1, unit)

    return tf.concat(ret, axis=1)

  def get_config(self, ):
    config = {'units': self.units,
              'num_experts_task': self.num_experts_task,
              'num_experts_share': self.num_experts_share,
              'num_tasks': self.num_tasks,
              'activation': self.activation,
              'output_activation': self.output_activation,
              'output_share': self.output_share,
              'use_bias': self.use_bias,
              'l2_reg': self.l2_reg,
              'seed': self.seed}
    base_config = super(CGC2, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    if self.output_share:
      return None, self.num_tasks + 1, self.units[-1]
    return None, self.num_tasks, self.units[-1]


class CGC(Layer):
  """ Input shape
        - 2D tensor with shape: ``(batch_size, units)`` or ``(batch_size, t + 1, units)``
        - Or list of ``[input, gate_input]``, gate_input with shape ``(batch_size, 1, gate_input_dim)``
      Output shape
        - 2D tensor with shape: ``(batch_size, t, units)`` or ``(batch_size, t + 1, units)``.
      Arguments
        - **units** : expert tower, feed forward network layer
        - **num_experts_task** : Positive integer, number of experts for each task.
        - **num_experts_share** : Positive integer, number of experts for share.
        - **num_tasks** : Positive integer, number of tasks.
        - **output_share** : Boolean, whether to output share expert.
        - **use_bias** : Boolean, whether to use weight bias.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
    """

  def __init__(self, num_tasks=2,
               num_experts_task=1,
               num_experts_share=2,
               units=[256],
               activation="relu",
               output_activation=None,
               output_share=False,
               use_bias=True,
               l2_reg=0, seed=1024, **kwargs):
    self.units = units
    self.num_experts_task = num_experts_task
    self.num_experts_share = num_experts_share
    self.num_tasks = num_tasks
    self.activation = activation
    self.output_activation = output_activation
    self.output_share = output_share
    self.use_bias = use_bias
    self.l2_reg = l2_reg
    self.seed = seed
    super(CGC, self).__init__(**kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, (list, tuple)):
      if len(input_shape[0]) == 3 and input_shape[0][1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %s, expect to be (batch_size, 1, dim) or (batch_size, %d, dim)" % (input_shape, self.num_tasks + 1))
      units = [input_shape[0][-1]] + self.units
    else:
      if len(input_shape) == 3 and input_shape[1] != self.num_tasks + 1:
        raise ValueError("Unexpected inputs shape %s, expect to be (batch_size, 1, dim) or (batch_size, %d, dim)" % (input_shape, self.num_tasks + 1))
      units = [input_shape[-1]] + self.units

    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share
    # expert
    self.experts = []
    self.experts_bias = []
    for n in range(num_experts):
      self.experts.append([self.add_weight(name=f'experts_weight_{n}_{i}',
                                           shape=(units[i], units[i + 1]),
                                           initializer=tf.initializers.glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg),
                                           trainable=True) for i in range(len(units) - 1)])
      self.experts_bias.append([self.add_weight(name=f'experts_bias_{n}_{i}',
                                                shape=(self.units[i],),
                                                initializer=tf.initializers.Zeros(),
                                                trainable=True) for i in range(len(self.units))])

    self.gating = [tf.keras.layers.Dense(self.num_experts_task + self.num_experts_share,
                                         name="gating_" + str(i),
                                         use_bias=False) for i in range(self.num_tasks)]

    if self.output_share:
      self.gating_share = tf.keras.layers.Dense(num_experts, name="gating_share", use_bias=False)

    self.activation_layers = [activation_layer(self.output_activation if i == len(self.units) - 1 and self.output_activation else self.activation)
                              for i in range(len(self.units))]

    # Be sure to call this somewhere!
    super(CGC, self).build(input_shape)

  def call(self, inputs):
    num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share

    x1 = None
    if isinstance(inputs, (list, tuple)):
      target_inputs, gate_input = inputs
      x1 = [gate_input] * (self.num_tasks + 1)
    else:
      target_inputs = inputs

    if K.ndim(target_inputs) == 2:
      x0 = tf.tile(tf.expand_dims(target_inputs, 1), [1, num_experts, 1])
      if not x1:
        x1 = [tf.expand_dims(target_inputs, axis=1)] * (self.num_tasks + 1)
    elif K.ndim(target_inputs) == 3:
      x0 = tf.repeat(target_inputs, [self.num_experts_task] * self.num_tasks + [self.num_experts_share], axis=1)
      if not x1:
        x1 = tf.split(target_inputs, num_or_size_splits=(self.num_tasks + 1), axis=1)
    else:
      raise ValueError("Unexpected inputs dims %d" % K.ndim(target_inputs))

    x0 = tf.split(x0, num_or_size_splits=num_experts, axis=1)  # n * (bs, 1, d)
    for n in range(num_experts):
      x0[n] = tf.squeeze(x0[n], axis=1)
      for i in range(len(self.units)):
        # for gpu which not support einsum
        x0[n] = tf.matmul(x0[n], self.experts[n][i])  # (bs, e)
        x0[n] = self.activation_layers[i](tf.add(x0[n], self.experts_bias[n][i]))

    ret = []
    for i in range(self.num_tasks):
      gating_score = tf.nn.softmax(self.gating[i](x1[i]))  # (bs, 1, num_experts_task + num_experts_share)
      output_of_experts = tf.stack(x0[i * self.num_experts_task:(i + 1) * self.num_experts_task] + x0[-self.num_experts_share:], axis=1)  # (bs, num_experts_task + num_experts_share, unit)
      ret.append(tf.matmul(gating_score, output_of_experts))  # (bs, 1, unit)

    if self.output_share:
      gating_score = tf.nn.softmax(self.gating_share(x1[-1]))  # (bs, 1, num_experts)
      ret.append(tf.matmul(gating_score, tf.stack(x0, axis=1)))  # (bs, 1, unit)

    return tf.concat(ret, axis=1)

  def get_config(self, ):
    config = {'units': self.units,
              'num_experts_task': self.num_experts_task,
              'num_experts_share': self.num_experts_share,
              'num_tasks': self.num_tasks,
              'activation': self.activation,
              'output_activation': self.output_activation,
              'output_share': self.output_share,
              'use_bias': self.use_bias,
              'l2_reg': self.l2_reg,
              'seed': self.seed}
    base_config = super(CGC, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    if self.output_share:
      return None, self.num_tasks + 1, self.units[-1]
    return None, self.num_tasks, self.units[-1]


class PLE(Layer):
  """ Input shape
      - 2D tensor with shape: ``(batch_size, units)``
    Output shape
      - 2D tensor with shape: ``(batch_size, t, units)``
    Arguments
      - **units** : expert tower, feed forward network layer.
      - **num_layers** : number of cgc layers.
      - **num_experts_task** : Positive integer, number of experts for each task.
      - **num_experts_share** : Positive integer, number of experts for share.
      - **num_tasks** : Positive integer, number of tasks.
      - **output_share** : Boolean, whether to output share expert.
      - **use_bias** : Boolean, whether to use weight bias.
      - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
      - **seed**: A Python integer to use as random seed.
    References
      - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
  """

  def __init__(self,
               num_tasks=2,
               num_layers=2,
               num_experts_task=1,
               num_experts_share=2,
               units=[256],
               activation="relu",
               output_activation=None,
               output_share=False,
               use_bias=True,
               l2_reg=0, seed=1024, **kwargs):
    self.units = units
    self.num_layers = num_layers
    self.num_experts_task = num_experts_task
    self.num_experts_share = num_experts_share
    self.num_tasks = num_tasks
    self.activation = activation
    self.output_activation = output_activation
    self.output_share = output_share
    self.use_bias = use_bias
    self.l2_reg = l2_reg
    self.seed = seed
    super(PLE, self).__init__(**kwargs)

  def build(self, input_shape):
    # cgc
    self.cgc_layers = [CGC(num_tasks=self.num_tasks,
                           num_experts_task=self.num_experts_task,
                           num_experts_share=self.num_experts_share,
                           units=self.units,
                           activation=self.activation,
                           output_activation=self.output_activation,
                           output_share=(i != self.num_layers - 1),
                           use_bias=self.use_bias,
                           l2_reg=self.l2_reg,
                           seed=self.seed) for i in range(self.num_layers)]

    # Be sure to call this somewhere!
    super(PLE, self).build(input_shape)

  def call(self, inputs):
    x0 = inputs
    for i in range(self.num_layers):
      x0 = self.cgc_layers[i](x0)
    return x0

  def get_config(self, ):
    config = {'units': self.units,
              'num_layers': self.num_layers,
              'num_experts_task': self.num_experts_task,
              'num_experts_share': self.num_experts_share,
              'num_tasks': self.num_tasks,
              'activation': self.activation,
              'output_activation': self.output_activation,
              'output_share': self.output_share,
              'use_bias': self.use_bias,
              'l2_reg': self.l2_reg,
              'seed': self.seed}
    base_config = super(PLE, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    return None, self.num_tasks, self.units[-1]
