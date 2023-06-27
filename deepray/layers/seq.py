# -*- codig:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.python.keras import backend as K

from .core import LocalActivationUnit


class Pooling(Layer):
  """
      input shape: (batch_size, seq_len, emb_dim)
      output shape: (batch_size, 1, emb_dim)
  """

  def __init__(self, combiner, **kwargs):
    self.combiner = combiner
    super(Pooling, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(Pooling, self).build(input_shape)

  def call(self, x, mask=None):
    if self.combiner == 'max':
      if mask is not None:
        x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.ones_like(x, dtype=tf.float32) * (-2 ** 32 + 1))
      return tf.reduce_max(x, axis=1, keepdims=True)

    # sum
    if self.combiner == 'sum':
      if mask is not None:
        x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.zeros_like(x, dtype=tf.float32))
      return tf.reduce_sum(x, axis=1, keepdims=True)

    if self.combiner == 'mean':
      if mask is not None:
        #                 logger.info(f"emb: {x}")
        mask = tf.expand_dims(mask, axis=-1)
        x = tf.where(mask, x, tf.zeros_like(x, dtype=tf.float32))
        y = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1, keepdims=True)
        #                 logger.info(f"mask: {mask}")
        #                 logger.info(f"pooling: {tf.math.divide_no_nan(tf.reduce_sum(x, axis=1, keepdims=True), y)}")
        return tf.math.divide_no_nan(tf.reduce_sum(x, axis=1, keepdims=True), y)
      return tf.reduce_mean(x, axis=1, keepdims=True)

    return x

  def get_config(self, ):
    config = {'combiner': self.combiner}
    base_config = super(Pooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class AttentionPooling(Layer):
  """The Attentional pooling operation used in DIN.
      Input shape
        - A list of three tensor: [query,keys,keys_length]
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **att_activation**: Activation function to use in attention net.
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **supports_masking**:If True,the input need to support masking.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

  def __init__(self, att_hidden_units=(64,), att_activation='elu', weight_normalization=True,  # 不设置为 True，会导致不收敛
               return_score=False, **kwargs):

    self.att_hidden_units = att_hidden_units
    self.att_activation = att_activation
    self.weight_normalization = weight_normalization
    self.return_score = return_score
    super(AttentionPooling, self).__init__(**kwargs)

  def build(self, input_shape):
    self.local_att = LocalActivationUnit(
      self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
    super(AttentionPooling, self).build(input_shape)

  def call(self, inputs, mask=None, training=None, **kwargs):
    queries, keys = inputs

    attention_score = self.local_att([queries, keys], training=training)
    outputs = tf.transpose(attention_score, (0, 2, 1))  # (B,1,T)
    # keys_shape = keys.get_shape().as_list()
    # outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) / tf.math.sqrt(keys_shape[2] * 1.0)

    if self.weight_normalization:
      paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    else:
      paddings = tf.zeros_like(outputs)

    if mask is not None:
      key_masks = tf.expand_dims(mask[-1], axis=1)
      outputs = tf.where(key_masks, outputs, paddings)

    if self.weight_normalization:
      outputs = tf.nn.softmax(outputs)

    if not self.return_score:
      outputs = tf.matmul(outputs, keys)

    outputs._uses_learning_phase = training is not None

    return outputs

  def compute_output_shape(self, input_shape):
    if self.return_score:
      return (None, 1, input_shape[1][1])
    else:
      return (None, 1, input_shape[0][-1])

  def compute_mask(self, inputs, mask):
    return None

  def get_config(self, ):

    config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
              'weight_normalization': self.weight_normalization,
              'return_score': self.return_score}
    base_config = super(AttentionPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SENETLayer(Layer):
  """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

  def __init__(self, reduction_ratio=3, seed=1024, use_bias=False, l2_reg=0, activation='relu', **kwargs):
    self.reduction_ratio = reduction_ratio
    self.l2_reg = l2_reg
    self.use_bias = use_bias
    self.activation = activation
    self.seed = seed
    super(SENETLayer, self).__init__(**kwargs)

  def build(self, input_shape):

    if not isinstance(input_shape, list) or len(input_shape) < 2:
      raise ValueError('A `AttentionalFM` layer should be called '
                       'on a list of at least 2 inputs')

    self.filed_size = len(input_shape)
    self.embedding_size = input_shape[0][-1]
    reduction_size = max(1, self.filed_size // self.reduction_ratio)

    self.dense_layers = [Dense(units,
                               use_bias=self.use_bias,
                               kernel_initializer=GlorotNormal(),
                               kernel_regularizer=l2(self.l2_reg) if self.l2_reg else None,
                               activation=self.activation)
                         for units in [reduction_size, self.filed_size]]

    # Be sure to call this somewhere!
    super(SENETLayer, self).build(input_shape)

  def call(self, inputs, training=None):

    if K.ndim(inputs[0]) != 3:
      raise ValueError(
        "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

    Zs = list()
    for i in inputs:
      Zs.append(tf.reduce_mean(i, axis=-1))
    Z = tf.concat(Zs, axis=-1)  # B * I
    for dense_layer in self.dense_layers:
      Z = dense_layer(Z)
    V = tf.split(Z, self.filed_size, axis=-1)

    Rs = list()
    for i, v in enumerate(V):
      Rs.append(tf.multiply(inputs[i], tf.expand_dims(v, axis=-1)))
    return Rs

  def compute_output_shape(self, input_shape):

    return input_shape

  def get_config(self, ):
    config = {'reduction_ratio': self.reduction_ratio,
              'seed': self.seed,
              'use_bias': self.use_bias,
              'l2_reg': self.l2_reg,
              'activation': self.activation}
    base_config = super(SENETLayer, self).get_config()
    base_config.update(config)
    return base_config


class AttentionPoolingSimplify6(Layer):
  def __init__(self, hidden_units=(32, 1), activation='sigmoid', use_bn=False, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.use_bn = use_bn

    self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]
    self.dense_layers = [Dense(self.hidden_units[i],
                               activation=None if i == len(self.hidden_units) - 1 else self.activation)
                         for i in range(len(self.hidden_units))]

    super(AttentionPoolingSimplify6, self).__init__(**kwargs)

  def build(self, input_shape):
    super(AttentionPoolingSimplify6, self).build(input_shape)

  def call(self, inputs, mask=None, **kwargs):
    query, keys = inputs  # (B, 1, size), (B, T, size)
    seq_len = tf.shape(keys)[1]

    queries = tf.tile(query, [1, seq_len, 1])  # (B, T, size)
    att_input = tf.concat([queries, keys, queries - keys], axis=-1)

    for i in range(len(self.hidden_units)):
      if self.use_bn:
        att_input = self.bn_layers[i](att_input)
      att_input = self.dense_layers[i](att_input)  # (B, T, 1)

    output = tf.reshape(att_input, [-1, 1, seq_len])  # (B, 1, T)

    if mask is not None:
      key_mask = tf.expand_dims(mask, axis=1)  # (B, T) -> (B, 1, T)
      paddings = tf.ones_like(output) * (-2 ** 32 + 1)  # (B, 1, T)
      output = tf.where(key_mask, output, paddings)  # (B, 1, T)

    output = tf.nn.softmax(output)  # (B, 1, T)
    output = tf.matmul(output, keys)  # (B, 1, size)

    return output

  def compute_output_shape(self, inputs):
    return (None, 1, inputs[0][-1])

  def get_config(self):
    config = {
      "hidden_units": self.hidden_units,
      "activation": self.activation,
      "use_bn": self.use_bn
    }
    base_config = super(AttentionPoolingSimplify6, self).get_config()

    return dict(list(base_config.items()) + list(config.items()))
