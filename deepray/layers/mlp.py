from copy import deepcopy
from typing import List

import tensorflow as tf
import tf_keras as keras


def extend_as_list(x, n):
  """This is a helper function to extend x as list, it will do:
  1. If x is a list, padding it to specified length n with None, if the length
  is less than n;
  2. If x is not a list, create a list with n elements x, please note that,
  these n elements are the same object, not a copy of x.
  """
  if isinstance(x, (list, tuple)):
    if len(x) < n:
      return x + [None] * (n - len(x))
    else:
      return x
  else:
    try:
      return [x if i == 0 else deepcopy(x) for i in range(n)]
    except:
      return [x] * n


class MLP(tf.keras.layers.Layer):
  """多层感知器(Multilayer Perceptron), 最经典的人工神经网络, 由一系列层叠起来的Dense层组成

  Args:
    output_dims (:obj:`List[int]`): 每一层的输出神经元个数
    activations (:obj:`List[tf.activation]`, `List[str]`, `tf.activation`, `str`): 激活函数, 可以用str表示, 也可以用TF中的activation
    initializers (:obj:`List[tf.initializer]`): kernel, 也就是W的初始化器, 是一个列表
    kernel_regularizer (:obj:`tf.regularizer`): kernel正侧化器
    use_bias (:obj:`bool`): 是否使用bias, 默认为True
    bias_regularizer (:obj:`tf.regularizer`): bias正侧化
    enable_batch_normalization (:obj:`bool`): 是否开启batch normalization, 如果开启, 会对输入数据, 及每个Dense Layer的输出匀做
                                              BatchNorm (最后一个Dense Layer除外).
    batch_normalization_momentum (:obj:`float`): BatchNorm中的动量因子
    batch_normalization_renorm (:obj:`bool`): 是否使用renorm, (论文可参考 https://arxiv.org/abs/1702.03275)
    batch_normalization_renorm_clipping (:obj:`bool`): renorm中的clipping, 具体请参考TF中的 `BatchNormalization`_
    batch_normalization_renorm_momentum (:obj:`float`): renorm中的momentum, 具体请参考TF中的 `BatchNormalization`_

  .. _BatchNormalization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

  """

  def __init__(
    self,
    hidden_units: List[int],
    name: str = "",
    activations=None,
    kernel_initializer=None,
    kernel_regularizer=None,
    use_bias=True,
    bias_regularizer=None,
    enable_batch_normalization=False,
    batch_normalization_momentum=0.99,
    batch_normalization_renorm=False,
    batch_normalization_renorm_clipping=None,
    batch_normalization_renorm_momentum=0.99,
    **kwargs,
  ):
    super(MLP, self).__init__(**kwargs)
    self.hidden_units = hidden_units
    self.prefix = name
    self.use_bias = use_bias
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    self.enable_batch_normalization = enable_batch_normalization
    self.batch_normalization_momentum = batch_normalization_momentum
    self.batch_normalization_renorm = batch_normalization_renorm
    self.batch_normalization_renorm_clipping = batch_normalization_renorm_clipping
    self.batch_normalization_renorm_momentum = batch_normalization_renorm_momentum
    self._stacked_layers = []
    self._n_layers = len(self.hidden_units)
    self._activations = []
    self._initializers = [tf.initializers.get(init) for init in extend_as_list(kernel_initializer, self._n_layers)]

    if activations is None:
      self._activations = [tf.keras.layers.Activation(activation="relu")] * (self._n_layers - 1) + [None]
    elif isinstance(activations, (list, tuple)):
      assert len(activations) == self._n_layers
      for act in activations:
        if act:
          self._activations.append(tf.keras.layers.Activation(activation=act))
        else:
          self._activations.append(None)
    else:
      self._activations = [
        tf.keras.layers.Activation(activation=activations) if i != self._n_layers - 1 else None
        for i in range(self._n_layers)
      ]

  def build(self, input_shape):
    if self.enable_batch_normalization:
      bn = keras.layers.BatchNormalization(
        momentum=self.batch_normalization_momentum,
        renorm=self.batch_normalization_renorm,
        renorm_clipping=self.batch_normalization_renorm_clipping,
        renorm_momentum=self.batch_normalization_renorm_momentum,
        name=f"{self.prefix}/BatchNorm/in",
      )
      self.trainable_weights.extend(bn.trainable_weights)
      self.non_trainable_weights.extend(bn.non_trainable_weights)
      self.add_loss(bn.losses)
      self._stacked_layers.append(bn)

    for layer_id, dim in enumerate(self.hidden_units):
      is_final_layer = layer_id == (self._n_layers - 1)
      dense = tf.keras.layers.Dense(
        name=f"dense_{self.prefix}{layer_id}",
        units=dim,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self._initializers[layer_id],
        bias_initializer=tf.initializers.zeros(),
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
      )
      self.trainable_weights.extend(dense.trainable_weights)
      self.non_trainable_weights.extend(dense.non_trainable_weights)
      self.add_loss(dense.losses)
      self._stacked_layers.append(dense)

      if not is_final_layer and self.enable_batch_normalization:
        bn = keras.layers.BatchNormalization(
          momentum=self.batch_normalization_momentum,
          renorm=self.batch_normalization_renorm,
          renorm_clipping=self.batch_normalization_renorm_clipping,
          renorm_momentum=self.batch_normalization_renorm_momentum,
          name=f"{self.prefix}/BatchNorm/out",
        )
        self.trainable_weights.extend(bn.trainable_weights)
        self.non_trainable_weights.extend(bn.non_trainable_weights)
        self.add_loss(bn.losses)
        self._stacked_layers.append(bn)

      if self._activations[layer_id] is not None:
        self._stacked_layers.append(self._activations[layer_id])

    super(MLP, self).build(input_shape)

  def call(self, input, **kwargs):
    input_t, output_t = input, None
    for layer in self._stacked_layers:
      output_t = layer(input_t)
      input_t = output_t
    return output_t

  def get_config(self):
    config = {
      "hidden_units": self.hidden_units,
      "name": self.prefix,
      "activations": [tf.keras.layers.Activation(activation=act) for act in self._activations],
      "initializers": [tf.initializers.serialize(init) for init in self._initializers],
      "enable_batch_normalization": self.enable_batch_normalization,
      "batch_normalization_momentum": self.batch_normalization_momentum,
      "use_bias": self.use_bias,
      "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
      "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
      "batch_normalization_renorm": self.batch_normalization_renorm,
      "batch_normalization_renorm_clipping": self.batch_normalization_renorm_clipping,
      "batch_normalization_renorm_momentum": self.batch_normalization_renorm_momentum,
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_layer(self, index: int):
    assert index < len(self._stacked_layers)
    return self._stacked_layers[index]
