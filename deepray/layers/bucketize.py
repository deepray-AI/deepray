import tensorflow as tf


class Bucketize(tf.keras.layers.Layer):
  def __init__(self, boundaries, **kwargs):
    self.boundaries = boundaries
    super(Bucketize, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(Bucketize, self).build(input_shape)

  def call(self, x, **kwargs):
    return tf.raw_ops.Bucketize(input=x, boundaries=self.boundaries)

  def get_config(
    self,
  ):
    config = {"boundaries": self.boundaries}
    base_config = super(Bucketize, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CategoryToIdLayer(tf.keras.layers.Layer):
  def __init__(self, vocabulary_list, values):
    self.vocabulary_list = vocabulary_list
    self.values = values
    init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.constant(self.vocabulary_list, dtype=tf.int64), values=tf.constant(self.values, dtype=tf.int64)
    )
    self.table = tf.lookup.StaticVocabularyTable(init, 1)
    super(CategoryToIdLayer, self).__init__()

  def call(self, inputs):
    id_tensor = self.table.lookup(inputs)
    return id_tensor


class HashLongToIdLayer(tf.keras.layers.Layer):
  def __init__(self, hash_bucket_size, mask=False):
    super(HashLongToIdLayer, self).__init__()
    self.hash_bucket_size = hash_bucket_size
    self.mask = mask

  def call(self, inputs):
    string_tensor = tf.strings.as_string(inputs)
    id_tensor = tf.strings.to_hash_bucket_fast(string_tensor, self.hash_bucket_size)
    if self.mask:
      mask_tensor = 1 - tf.cast(inputs <= 0, tf.int64)
      return id_tensor * mask_tensor
    else:
      return id_tensor


class Hash(tf.keras.layers.Layer):
  def __init__(self, hash_size: int, mask: bool = False, **kwargs):
    """
    Initializes the Hash object with a hash_size and an optional mask argument.
    :param hash_size: The size of the hash bucket.
    :param mask: If True, masks out values less than or equal to 0 in the output tensor.
    """
    super().__init__(**kwargs)
    if not isinstance(hash_size, int):
      raise TypeError(f"hash_size must be an integer, your input is {hash_size}")
    self.hash_size = hash_size
    self.mask = mask

  def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    """
    Applies the hash function to the input tensor and returns the result.
    :param inputs: The input tensor to apply the hash function to.
    :return: The result of applying the hash function to the input tensor.
    """
    output_tensor = tf.math.floormod(inputs, self.hash_size)
    if self.mask:
      mask_tensor = 1 - tf.cast(output_tensor <= 0, tf.int64)
      return output_tensor * mask_tensor
    else:
      return output_tensor


class NumericaBucketIdLayer(tf.keras.layers.Layer):
  def __init__(self, bucket_boundaries):
    self.bucket_boundaries = bucket_boundaries
    super(NumericaBucketIdLayer, self).__init__()

  def call(self, inputs, *args, **kwargs):
    id_tensor = tf.raw_ops.Bucketize(input=inputs, boundaries=self.bucket_boundaries)
    return id_tensor
