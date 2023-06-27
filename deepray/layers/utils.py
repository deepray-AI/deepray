# -*- coding:utf-8 -*-
import tensorflow as tf


# from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, StringLookup


# tf 2.4.*以下版本 StringLookup 可能会导致 save_model 时 SegmentationFault
# tf v2.5.*再试, https://github.com/tensorflow/tensorflow/issues/45449
# class VocabLookupNew(tf.keras.layers.Layer):
#
#     def __init__(self, vocabs, vtype='int64', num_oov_buckets=0, **kwargs):
#         self.vocabs = vocabs
#         self.vtype = vtype
#         self.num_oov_buckets = num_oov_buckets
#         super(VocabLookupNew, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Be sure to call this somewhere!
#         super(VocabLookupNew, self).build(input_shape)
#         vocabs = tf.constant(self.vocabs, dtype=self.vtype)
#         if vocabs.dtype != tf.string:
#             vocabs = tf.cast(vocabs, dtype=tf.int64)
#             self.table = IntegerLookup(num_oov_indices=self.num_oov_buckets)
#         else:
#             self.table = StringLookup(num_oov_indices=self.num_oov_buckets)
#         self.table.adapt(vocabs)
#
#     def call(self, x):
#         return self.table(x)
#
#     def get_config(self, ):
#         config = {'vocabs': self.vocabs, 'vtype': self.vtype, 'num_oov_buckets': self.num_oov_buckets}
#         base_config = super(VocabLookupNew, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class VocabLookup(tf.keras.layers.Layer):

  def __init__(self, vocabs, vtype='int64', num_oov_buckets=0, **kwargs):
    self.vocabs = vocabs
    self.vtype = vtype
    self.num_oov_buckets = num_oov_buckets
    super(VocabLookup, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(VocabLookup, self).build(input_shape)
    vocabs = tf.constant(self.vocabs, dtype=self.vtype)
    if vocabs.dtype != tf.string:
      vocabs = tf.cast(vocabs, dtype=tf.int64)

    oov = self.num_oov_buckets
    if oov <= 0:
      oov = 1
    self.table = tf.lookup.StaticVocabularyTable(
      num_oov_buckets=oov,
      initializer=tf.lookup.KeyValueTensorInitializer(
        keys=vocabs,
        values=tf.range(len(self.vocabs), dtype=tf.int64)
      )
    )

  def call(self, x):
    ids = self.table.lookup(x)
    if self.num_oov_buckets <= 0:
      oov = tf.cast(tf.fill(tf.shape(ids), -1), tf.int64)
      ids = tf.where(ids < len(self.vocabs), ids, oov)
    return ids

  def get_config(self, ):
    config = {'vocabs': self.vocabs, 'vtype': self.vtype, 'num_oov_buckets': self.num_oov_buckets}
    base_config = super(VocabLookup, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class PaddingTrim(tf.keras.layers.Layer):

  def __init__(self, padding, **kwargs):
    self.padding = padding
    super(PaddingTrim, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(PaddingTrim, self).build(input_shape)

  def call(self, column):
    if column.shape[-1] == 1:
      return column

    dtype = column.dtype
    column = tf.RaggedTensor.from_tensor(column, padding=tf.cast(self.padding, dtype=dtype))
    marker = tf.cast(tf.fill([column.nrows(), 1], self.padding), dtype=dtype)
    return tf.concat([column, marker], axis=1)

  def get_config(self, ):
    config = {'padding': self.padding}
    base_config = super(PaddingTrim, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Hash(tf.keras.layers.Layer):
  """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

  def __init__(self, num_buckets, mask_zero=False, **kwargs):
    self.num_buckets = num_buckets
    self.mask_zero = mask_zero
    super(Hash, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(Hash, self).build(input_shape)

  def call(self, x, mask=None):

    if x.dtype != tf.string:
      zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
      x = tf.as_string(x, )
    else:
      zero = tf.as_string(tf.zeros([1], dtype='int32'))

    num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
    try:
      hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                             name=None)  # weak hash
    except:
      hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                              name=None)  # weak hash
    if self.mask_zero:
      mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
      hash_x = (hash_x + 1) * mask

    return hash_x

  def get_config(self, ):
    config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
    base_config = super(Hash, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

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
      linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
    elif self.mode == 1:
      dense_input = inputs
      fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
      linear_logit = fc
    else:
      sparse_input, dense_input = inputs
      fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
      linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
    if self.use_bias:
      linear_logit += self.bias

    return linear_logit

  def compute_output_shape(self, input_shape):
    return (None, 1)

  def compute_mask(self, inputs, mask):
    return None

  def get_config(self, ):
    config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
    base_config = super(Linear, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SimpleReWeight(tf.keras.layers.Layer):

  def __init__(self, max_count, raw_weights, **kwargs):
    self.max_count = max_count
    self.raw_weights = raw_weights
    super(SimpleReWeight, self).__init__(**kwargs)

  def call(self, inputs):
    status_output = tf.reduce_sum(inputs, axis=-1, keepdims=True)
    weights = tf.constant(self.raw_weights, dtype=tf.float32)
    # return tf.map_fn(lambda s: weights if s >= 1 else 1.0 - weights, status_output)
    return tf.where(status_output >= self.max_count, weights, 1 - weights)

  def get_config(self, ):
    config = {'raw_weights': self.raw_weights, 'max_count': self.max_count}
    base_config = super(SimpleReWeight, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SimpleCountReWeight(tf.keras.layers.Layer):

  def __init__(self, max_count, raw_weights, **kwargs):
    self.max_count = max_count
    self.raw_weights = raw_weights
    super(SimpleCountReWeight, self).__init__(**kwargs)

  def call(self, inputs):
    status_output = tf.reduce_sum(inputs, axis=-1, keepdims=True)
    weights = tf.constant(self.raw_weights, dtype=tf.float32)
    addf_factor = tf.cast(tf.maximum(self.max_count - status_output, 0) / self.max_count, dtype=tf.float32)
    # addf_factors = tf.cast(tf.concat([1 - addf_factor, addf_factor], axis=-1), dtype=tf.float32)
    addf_factors = tf.concat([tf.ones_like(addf_factor), addf_factor], axis=-1)
    addf_weight = tf.reduce_sum(addf_factors * weights, axis=-1, keepdims=True)
    return tf.concat([addf_weight, 1 - addf_weight], axis=-1)

  def get_config(self, ):
    config = {'raw_weights': self.raw_weights, 'max_count': self.max_count}
    base_config = super(SimpleCountReWeight, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
