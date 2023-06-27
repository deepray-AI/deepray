# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization
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


class DNN(Layer):
  def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
               output_activation='linear',
               seed=1024, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.output_activation = output_activation
    self.seed = seed

    super(DNN, self).__init__(**kwargs)

  def build(self, input_shape):
    activations = [self.activation] * (len(self.hidden_units) - 1) + [self.output_activation if self.output_activation else self.activation]

    self.dense_layers = [Dense(self.hidden_units[i],
                               kernel_initializer=GlorotNormal(),
                               kernel_regularizer=l2(self.l2_reg) if self.l2_reg else None,
                               activation=activations[i])
                         for i in range(len(self.hidden_units))]

    if self.use_bn:
      self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

    if self.dropout_rate:
      self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                             range(len(self.hidden_units))]

    super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, training=None):

    deep_input = inputs

    for i in range(len(self.hidden_units)):
      fc = self.dense_layers[i](deep_input)

      if self.use_bn:
        fc = self.bn_layers[i](fc, training=training)

      if self.dropout_rate:
        fc = self.dropout_layers[i](fc, training=training)

      deep_input = fc

    return deep_input

  def compute_output_shape(self, input_shape):
    if len(self.hidden_units) > 0:
      shape = input_shape[:-1] + (self.hidden_units[-1],)
    else:
      shape = input_shape

    return tuple(shape)

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
              'output_activation': self.output_activation, 'seed': self.seed}
    base_config = super(DNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class DNN2(Layer):
  def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
               output_activation='linear',
               seed=1024, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.output_activation = output_activation
    self.seed = seed

    super(DNN2, self).__init__(**kwargs)

  def build(self, input_shape):
    # if len(self.hidden_units) == 0:
    #     raise ValueError("hidden_units is empty")
    input_size = input_shape[-1]
    hidden_units = [int(input_size)] + list(self.hidden_units)
    self.kernels = [self.add_weight(name='kernel' + str(i),
                                    shape=(
                                      hidden_units[i], hidden_units[i + 1]),
                                    initializer=tf.initializers.glorot_normal(seed=self.seed),
                                    regularizer=l2(self.l2_reg),
                                    trainable=True) for i in range(len(self.hidden_units))]
    self.bias = [self.add_weight(name='bias' + str(i),
                                 shape=(self.hidden_units[i],),
                                 initializer=tf.initializers.Zeros(),
                                 trainable=True) for i in range(len(self.hidden_units))]
    if self.use_bn:
      self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

    self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                           range(len(self.hidden_units))]

    self.activation_layers = [activation_layer(self.output_activation if i == len(self.hidden_units) - 1 and self.output_activation else self.activation)
                              for i in range(len(self.hidden_units))]

    super(DNN2, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, training=None):

    deep_input = inputs

    for i in range(len(self.hidden_units)):
      fc = tf.nn.bias_add(tf.tensordot(
        deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

      if self.use_bn:
        fc = self.bn_layers[i](fc, training=training)

      fc = self.activation_layers[i](fc)

      fc = self.dropout_layers[i](fc, training=training)
      deep_input = fc

    return deep_input

  def compute_output_shape(self, input_shape):
    if len(self.hidden_units) > 0:
      shape = input_shape[:-1] + (self.hidden_units[-1],)
    else:
      shape = input_shape

    return tuple(shape)

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
              'output_activation': self.output_activation, 'seed': self.seed}
    base_config = super(DNN2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class DNN3(Layer):
  def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
               output_activation='linear',
               seed=1024, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.output_activation = output_activation
    self.seed = seed

    super(DNN3, self).__init__(**kwargs)

  def build(self, input_shape):
    activations = [self.activation] * (len(self.hidden_units) - 1) + [self.output_activation if self.output_activation else self.activation]

    self.dense_layers = [Dense(self.hidden_units[i],
                               kernel_initializer=GlorotNormal(),
                               kernel_regularizer=l2(self.l2_reg) if self.l2_reg else None,
                               activation=activations[i],
                               trainable=False)
                         for i in range(len(self.hidden_units))]

    if self.use_bn:
      self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

    if self.dropout_rate:
      self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                             range(len(self.hidden_units))]

    super(DNN3, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, training=None):

    deep_input = inputs

    for i in range(len(self.hidden_units)):
      fc = self.dense_layers[i](deep_input)

      if self.use_bn:
        fc = self.bn_layers[i](fc, training=training)

      if self.dropout_rate:
        fc = self.dropout_layers[i](fc, training=training)

      deep_input = fc

    return deep_input

  def compute_output_shape(self, input_shape):
    if len(self.hidden_units) > 0:
      shape = input_shape[:-1] + (self.hidden_units[-1],)
    else:
      shape = input_shape

    return tuple(shape)

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
              'output_activation': self.output_activation, 'seed': self.seed}
    base_config = super(DNN3, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FFM(Layer):
  """
        FFM Config: field_a, field_b
        input_shape: (batch_size, len_a, fea_len * len_b), (batch_size, len_b, fea_len * len_a)
        output_shape: (batch_size, 1)
    """

  def __init__(self, fea_len, **kwargs):
    self.fea_len = fea_len
    super(FFM, self).__init__(**kwargs)

  def build(self, input_shape):
    super(FFM, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs):
    fea_a, fea_b = inputs
    fea_len = self.fea_len

    # reshape to (batch_size, len_b, len_a, fea_len)
    fea_b = tf.reshape(fea_b, [-1, tf.shape(fea_b)[1], tf.shape(fea_b)[2] // fea_len, fea_len])
    # transpose to (batch_size, len_a, len_b, fea_len)
    fea_b = tf.transpose(fea_b, [0, 2, 1, 3])
    # same with with fea_a: (batch_size, len_a, fea_len * len_b)
    fea_b = tf.reshape(fea_b, [tf.shape(fea_b)[0], tf.shape(fea_b)[1], -1])

    return tf.expand_dims(tf.reduce_sum(fea_a * fea_b, axis=[1, 2]), axis=-1)

  def compute_output_shape(self, input_shape):
    return None, 1

  def get_config(self):
    config = {'fea_len': self.fea_len}
    base_config = super(FFM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FM(Layer):
  """
        input_shape: (batch_size, T, embedding_size)
        output_shape: (batch_size, 1)
    """

  def __init__(self, **kwargs):
    super(FM, self).__init__(**kwargs)

  def build(self, input_shape):
    super(FM, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs):
    concated_embeds_value = inputs

    square_of_sum = tf.square(tf.reduce_sum(
      concated_embeds_value, axis=1, keepdims=True))
    sum_of_square = tf.reduce_sum(
      concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

    return cross_term

  def compute_output_shape(self, input_shape):
    return None, 1


class Linear(Layer):
  """
        input_shape: (batch_size, N) or [(batch_size, M), (batch_size, N)]
        output_shape: (batch_size, 1)
        mode:
            0, input is sparse weight, output = sum(input) + b
            1, input is dense, output = w * input + b
            2, input is sparse and dense, output = sum(sparse_input) + w * dense_input + b
    """

  def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):

    self.l2_reg = l2_reg
    # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
    if mode not in [0, 1, 2]:
      raise ValueError("mode must be 0,1 or 2")
    self.mode = mode
    self.use_bias = use_bias
    self.seed = seed
    super(Linear, self).__init__(**kwargs)

  def build(self, input_shape):
    if self.use_bias:
      self.bias = self.add_weight(name='linear_bias',
                                  shape=(1,),
                                  initializer=tf.keras.initializers.Zeros(),
                                  trainable=True)
    if self.mode == 1:
      self.kernel = self.add_weight(
        'linear_kernel',
        shape=[int(input_shape[-1]), 1],
        initializer=tf.keras.initializers.glorot_normal(self.seed),
        regularizer=tf.keras.regularizers.l2(self.l2_reg),
        trainable=True)
    elif self.mode == 2:
      self.kernel = self.add_weight(
        'linear_kernel',
        shape=[int(input_shape[1][-1]), 1],
        initializer=tf.keras.initializers.glorot_normal(self.seed),
        regularizer=tf.keras.regularizers.l2(self.l2_reg),
        trainable=True)

    super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs):
    if self.mode == 0:
      sparse_input = inputs
      linear_logit = tf.reduce_sum(sparse_input, axis=-1, keepdims=True)
    elif self.mode == 1:
      dense_input = inputs
      fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
      linear_logit = fc
    else:
      sparse_input, dense_input = inputs
      fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
      linear_logit = tf.reduce_sum(sparse_input, axis=-1, keepdims=True) + fc
    if self.use_bias:
      linear_logit += self.bias

    return linear_logit

  def compute_output_shape(self, input_shape):
    return (None, 1)

  def get_config(self, ):
    config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
    base_config = super(Linear, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CrossNet(Layer):
  """
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

  def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
    self.layer_num = layer_num
    self.parameterization = parameterization
    self.l2_reg = l2_reg
    self.seed = seed
    print('CrossNet parameterization:', self.parameterization)
    super(CrossNet, self).__init__(**kwargs)

  def build(self, input_shape):

    if len(input_shape) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

    dim = int(input_shape[-1])
    if self.parameterization == 'vector':
      self.kernels = [self.add_weight(name='kernel' + str(i),
                                      shape=(dim, 1),
                                      initializer=tf.initializers.glorot_normal(
                                        seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      trainable=True) for i in range(self.layer_num)]
    elif self.parameterization == 'matrix':
      self.kernels = [self.add_weight(name='kernel' + str(i),
                                      shape=(dim, dim),
                                      initializer=tf.initializers.glorot_normal(
                                        seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      trainable=True) for i in range(self.layer_num)]
    else:  # error
      raise ValueError("parameterization should be 'vector' or 'matrix'")
    self.bias = [self.add_weight(name='bias' + str(i),
                                 shape=(dim, 1),
                                 initializer=tf.initializers.Zeros(),
                                 trainable=True) for i in range(self.layer_num)]
    # Be sure to call this somewhere!
    super(CrossNet, self).build(input_shape)

  def call(self, inputs):
    if K.ndim(inputs) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

    x_0 = tf.expand_dims(inputs, axis=2)
    x_l = x_0
    for i in range(self.layer_num):
      if self.parameterization == 'vector':
        xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))  # (dim, 1) * (b, dim, 1) -> (b, 1, 1)
        dot_ = tf.matmul(x_0, xl_w)  # (b, dim, 1) * (b, 1, 1) -> (b, dim, 1)
        x_l = dot_ + self.bias[i] + x_l
      elif self.parameterization == 'matrix':
        xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
        dot_ = xl_w + self.bias[i]  # W * xi + b
        x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
      else:  # error
        raise ValueError("parameterization should be 'vector' or 'matrix'")
    x_l = tf.squeeze(x_l, axis=2)
    return x_l

  def get_config(self, ):

    config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
              'l2_reg': self.l2_reg, 'seed': self.seed}
    base_config = super(CrossNet, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    return input_shape


class CrossNetMix(Layer):
  """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

  def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, seed=1024, **kwargs):
    self.low_rank = low_rank
    self.num_experts = num_experts
    self.layer_num = layer_num
    self.l2_reg = l2_reg
    self.seed = seed
    super(CrossNetMix, self).__init__(**kwargs)

  def build(self, input_shape):

    if len(input_shape) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

    dim = int(input_shape[-1])

    # U: (dim, low_rank)
    self.U_list = [self.add_weight(name='U_list' + str(i),
                                   shape=(self.num_experts, dim, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]
    # V: (dim, low_rank)
    self.V_list = [self.add_weight(name='V_list' + str(i),
                                   shape=(self.num_experts, dim, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]
    # C: (low_rank, low_rank)
    self.C_list = [self.add_weight(name='C_list' + str(i),
                                   shape=(self.num_experts, self.low_rank, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]

    self.gating = [tf.keras.layers.Dense(1, use_bias=False) for i in range(self.num_experts)]

    self.bias = [self.add_weight(name='bias' + str(i),
                                 shape=(dim, 1),
                                 initializer=tf.initializers.Zeros(),
                                 trainable=True) for i in range(self.layer_num)]
    # Be sure to call this somewhere!
    super(CrossNetMix, self).build(input_shape)

  def call(self, inputs):
    if K.ndim(inputs) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

    x_0 = tf.expand_dims(inputs, axis=2)
    x_l = x_0
    for i in range(self.layer_num):
      output_of_experts = []
      gating_score_of_experts = []
      for expert_id in range(self.num_experts):
        # (1) G(x_l)
        # compute the gating score by x_l
        gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

        # (2) E(x_l)
        # project the input x_l to $\mathbb{R}^{r}$
        v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

        # nonlinear activation in low rank space
        v_x = tf.nn.tanh(v_x)
        v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
        v_x = tf.nn.tanh(v_x)

        # project back to $\mathbb{R}^{d}$
        uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

        dot_ = uv_x + self.bias[i]
        dot_ = x_0 * dot_  # Hadamard-product

        output_of_experts.append(tf.squeeze(dot_, axis=2))

      # (3) mixture of low-rank experts
      output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
      gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
      moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
      x_l = moe_out + x_l  # (bs, dim, 1)
    x_l = tf.squeeze(x_l, axis=2)
    return x_l

  def get_config(self, ):

    config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
              'l2_reg': self.l2_reg, 'seed': self.seed}
    base_config = super(CrossNetMix, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    return input_shape


class CrossNetMix2(Layer):
  """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

  def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, seed=1024, **kwargs):
    self.low_rank = low_rank
    self.num_experts = num_experts
    self.layer_num = layer_num
    self.l2_reg = l2_reg
    self.seed = seed
    super(CrossNetMix2, self).__init__(**kwargs)

  def build(self, input_shape):

    if len(input_shape) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

    dim = int(input_shape[-1])

    # U: (dim, low_rank)
    self.U_list = [self.add_weight(name='U_list' + str(i),
                                   shape=(self.num_experts, dim, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]
    # V: (dim, low_rank)
    self.V_list = [self.add_weight(name='V_list' + str(i),
                                   shape=(self.num_experts, dim, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]
    # C: (low_rank, low_rank)
    self.C_list = [self.add_weight(name='C_list' + str(i),
                                   shape=(self.num_experts, self.low_rank, self.low_rank),
                                   initializer=tf.initializers.glorot_normal(
                                     seed=self.seed),
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for i in range(self.layer_num)]

    self.gating = tf.keras.layers.Dense(self.num_experts, use_bias=False)

    self.bias = [self.add_weight(name='bias' + str(i),
                                 shape=(dim,),
                                 initializer=tf.initializers.Zeros(),
                                 trainable=True) for i in range(self.layer_num)]
    # Be sure to call this somewhere!
    super(CrossNetMix2, self).build(input_shape)

  def call(self, inputs):
    if K.ndim(inputs) != 2:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

    x_l = inputs  # (bs, dim)
    x_0 = tf.tile(tf.expand_dims(inputs, 1), [1, self.num_experts, 1])  # (bs, num_experts, dim)
    for i in range(self.layer_num):
      # project the input x_l to $\mathbb{R}^{r}$
      v_x = tf.einsum('bd,edr->ber', x_l, self.V_list[i])  # (bs, num_experts, low_rank)
      # nonlinear activation in low rank space
      v_x = tf.nn.tanh(v_x)

      v_x = tf.einsum('eij,bej->bei', self.C_list[i], v_x)  # (bs, num_experts, low_rank)
      v_x = tf.nn.tanh(v_x)

      # project back to $\mathbb{R}^{d}$
      uv_x = tf.einsum('edr,ber->bed', self.U_list[i], v_x)  # (bs, num_experts, dim)

      dot_ = uv_x + self.bias[i]
      output_of_experts = x_0 * dot_  # Hadamard-product

      gating_score_of_experts = tf.expand_dims(self.gating(x_l), axis=1)  # (bs, 1, num_experts)
      moe_out = tf.matmul(tf.nn.softmax(gating_score_of_experts, 2), output_of_experts)
      x_l = tf.squeeze(moe_out, axis=1) + x_l  # (bs, dim)
    return x_l

  def get_config(self, ):

    config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
              'l2_reg': self.l2_reg, 'seed': self.seed}
    base_config = super(CrossNetMix2, self).get_config()
    base_config.update(config)
    return base_config

  def compute_output_shape(self, input_shape):
    return input_shape


class LocalActivationUnit(Layer):
  """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.
      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``
      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.
      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **activation**: Activation function to use in attention net.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.
        - **seed**: A Python integer to use as random seed.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

  def __init__(self, hidden_units=(64,), activation='elu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
               **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.seed = seed
    super(LocalActivationUnit, self).__init__(**kwargs)
    self.supports_masking = True

  def build(self, input_shape):

    if not isinstance(input_shape, list) or len(input_shape) != 2:
      raise ValueError('A `LocalActivationUnit` layer should be called '
                       'on a list of 2 inputs')

    if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
      raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
        len(input_shape[0]), len(input_shape[1])))

    if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
      raise ValueError('A `LocalActivationUnit` layer requires '
                       'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                       'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
    size = 4 * int(input_shape[0][-1]) if len(self.hidden_units) == 0 else self.hidden_units[-1]
    self.kernel = self.add_weight(shape=(size, 1),
                                  initializer=tf.initializers.glorot_normal(seed=self.seed),
                                  name="kernel")
    self.bias = self.add_weight(shape=(1,), initializer=tf.initializers.Zeros(), name="bias")
    self.dnn = DNN2(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

    super(LocalActivationUnit, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, training=None, **kwargs):

    query, keys = inputs

    keys_len = keys.get_shape()[1]
    queries = K.repeat_elements(query, keys_len, 1)

    att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

    att_out = self.dnn(att_input, training=training)

    attention_score = tf.nn.bias_add(tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)

    return attention_score

  def compute_output_shape(self, input_shape):
    return input_shape[1][:2] + (1,)

  def compute_mask(self, inputs, mask):
    return mask

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
    base_config = super(LocalActivationUnit, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
