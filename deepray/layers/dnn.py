import tensorflow as tf
from typing import List, Union


class DNN(tf.keras.layers.Layer):
  def __init__(self, hidden_units: List[int],
               activations: List[Union[str, callable]] = ['relu'],
               use_bn: bool = False,
               use_ln: bool = False,
               name: str = '',
               **kwargs):
    super().__init__(**kwargs)
    if len(activations) == 1 and len(hidden_units) != len(activations):
      self.activations = activations * len(hidden_units)
    else:
      self.activations = activations

    self.hidden = hidden_units
    self.use_bn = use_bn
    self.use_ln = use_ln
    self.prefix = name

  def build(self, input_shape):
    self.dense_kernel = [tf.keras.layers.Dense(units=num_hidden_units, name=f"dense_{self.prefix}{layer_id}") for layer_id, num_hidden_units in enumerate(self.hidden)]
    self.activation_layers = [tf.keras.layers.Activation(activation=activation) for layer_id, activation in enumerate(self.activations)]
    if self.use_bn:
      self.bn_layers = [tf.keras.layers.BatchNormalization(name='bn_' + str(i)) for i in range(len(self.hidden))]
    if self.use_ln:
      self.ln_layers = [tf.keras.layers.LayerNormalization(name='ln_' + str(i)) for i in range(len(self.hidden))]

  def call(self, inputs: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
    x = inputs
    for layer_id, _ in enumerate(self.hidden):
      x = self.apply_kernel(layer_id, x, training)
    return x

  def apply_kernel(self, layer_id: int, x, training):
    x = self.dense_kernel[layer_id](x)
    if self.use_bn:
      x = self.bn_layers[layer_id](x, training=training)

    if self.activation_layers[layer_id]:
      x = self.activation_layers[layer_id](x)

    if self.use_ln:
      x = self.ln_layers[layer_id](x, training=training)
    return x