import tensorflow as tf


class Pooling(tf.keras.layers.Layer):
  """
  input shape: (batch_size, seq_len, emb_dim)
  output shape: (batch_size, 1, emb_dim)
  """

  def __init__(self, combiner, **kwargs):
    self.combiner = combiner
    super(Pooling, self).__init__(**kwargs)
    self.supports_masking = True

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(Pooling, self).build(input_shape)

  def call(self, x, mask=None, **kwargs):
    if self.combiner == "max":
      if mask is not None:
        x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.ones_like(x, dtype=tf.float32) * (-(2**32) + 1))
      return tf.reduce_max(x, axis=1, keepdims=True)

    # sum
    if self.combiner == "sum":
      if mask is not None:
        x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.zeros_like(x, dtype=tf.float32))
      return tf.reduce_sum(x, axis=1, keepdims=True)

    if self.combiner == "mean":
      if mask is not None:
        mask = tf.expand_dims(mask, axis=-1)
        x = tf.where(mask, x, tf.zeros_like(x, dtype=tf.float32))
        y = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1, keepdims=True)
        return tf.math.divide_no_nan(tf.reduce_sum(x, axis=1, keepdims=True), y)
      return tf.reduce_mean(x, axis=1, keepdims=True)

    return x

  def get_config(
    self,
  ):
    config = {"combiner": self.combiner}
    base_config = super(Pooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
