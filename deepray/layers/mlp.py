import tensorflow as tf
from typing import List, Union


class MLP(tf.keras.layers.Layer):

  def __init__(
      self,
      hidden_units: List[int],
      activations: List[Union[str, callable]] = None,
      use_bn: bool = False,
      use_ln: bool = False,
      dropout_rate: float = 0,
      name: str = '',
      **kwargs
  ):
    super().__init__(**kwargs)
    if activations is not None and len(activations) == 1 and len(hidden_units) != len(activations):
      self.activations = activations * len(hidden_units)
    else:
      self.activations = activations

    self.hidden = hidden_units
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.use_ln = use_ln
    self.prefix = name

  def build(self, input_shape):
    self.dense_kernel = [
        tf.keras.layers.Dense(units=num_hidden_units, name=f"dense_{self.prefix}{layer_id}")
        for layer_id, num_hidden_units in enumerate(self.hidden)
    ]
    if self.activations:
      self.activation_layers = [tf.keras.layers.Activation(activation=activation) for activation in self.activations]
    if self.use_bn:
      self.bn_layers = [tf.keras.layers.BatchNormalization(name='bn_' + str(i)) for i in range(len(self.hidden))]
    if self.use_ln:
      self.ln_layers = [tf.keras.layers.LayerNormalization(name='ln_' + str(i)) for i in range(len(self.hidden))]

    if self.dropout_rate:
      self.dropout_layers = [
          tf.keras.layers.Dropout(self.dropout_rate, seed=1024 + i) for i in range(len(self.hidden_units))
      ]

  def call(self, inputs: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
    x = inputs
    for layer_id, _ in enumerate(self.hidden):
      x = self.apply_kernel(layer_id, x, training)
    return x

  def apply_kernel(self, layer_id: int, x, training):
    x = self.dense_kernel[layer_id](x)
    if self.use_bn:
      x = self.bn_layers[layer_id](x, training=training)

    if self.activations:
      x = self.activation_layers[layer_id](x)

    if self.use_ln:
      x = self.ln_layers[layer_id](x, training=training)

    if self.dropout_rate:
      x = self.dropout_layers[layer_id](x, training=training)
    return x

  def get_config(self,):
    config = {
        'hidden_units': self.hidden,
        'activations': self.activations,
        'use_bn': self.use_bn,
        'use_ln': self.use_ln,
        'dropout_rate': self.dropout_rate,
        'name': self.prefix
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
