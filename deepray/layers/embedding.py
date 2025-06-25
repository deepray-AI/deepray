# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Embedding layer."""

import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from packaging.version import parse

if parse(tf.__version__.replace("-tf", "+tf")) < parse("2.11"):
  from keras import backend
  from keras import constraints
  from keras import initializers
  from keras import regularizers
  from keras.dtensor import utils
  from keras.engine import base_layer_utils
  from keras.engine.base_layer import Layer
  from keras.utils import tf_utils
elif parse(tf.__version__) > parse("2.16.0"):
  from tf_keras.src import backend
  from tf_keras.src import constraints
  from tf_keras.src import initializers
  from tf_keras.src import regularizers
  from tf_keras.src.dtensor import utils
  from tf_keras.src.engine import base_layer_utils
  from tf_keras.src.engine.base_layer import Layer
  from tf_keras.src.utils import tf_utils
else:
  from keras.src import backend
  from keras.src import constraints
  from keras.src import initializers
  from keras.src import regularizers
  from keras.src.dtensor import utils
  from keras.src.engine import base_layer_utils
  from keras.src.engine.base_layer import Layer
  from keras.src.utils import tf_utils

import tf_keras as keras
import deepray as dp
from deepray.layers.bucketize import Hash


def get_variable_path(checkpoint_path, name, i=0):
  tokens = name.split("/")
  tokens = [t for t in tokens if "model_parallel" not in t and "data_parallel" not in t]
  name = "_".join(tokens)
  name = name.replace(":", "_")
  filename = name + f"_part{i}" + ".npy"
  return os.path.join(checkpoint_path, filename)


# @keras_export("keras.layers.Embedding")
class Embedding(Layer):
  """Turns positive integers (indexes) into dense vectors of fixed size.

  e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

  This layer can only be used on positive integer inputs of a fixed range. The
  `tf.keras.layers.TextVectorization`, `keras.layers.StringLookup`,
  and `tf.keras.layers.IntegerLookup` preprocessing layers can help prepare
  inputs for an `Embedding` layer.

  This layer accepts `tf.Tensor`, `tf.RaggedTensor` and `tf.SparseTensor`
  input.

  Example:

  >>> model = tf.keras.Sequential()
  >>> model.add(dp.layers.Embedding(1000, 64, input_length=10))
  >>> # The model will take as input an integer matrix of size (batch,
  >>> # input_length), and the largest integer (i.e. word index) in the input
  >>> # should be no larger than 999 (vocabulary size).
  >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
  >>> # dimension.
  >>> input_array = np.random.randint(1000, size=(32, 10))
  >>> model.compile('rmsprop', 'mse')
  >>> output_array = model.predict(input_array)
  >>> print(output_array.shape)
  (32, 10, 64)

  Args:
    vocabulary_size: Integer. Size of the vocabulary,
      i.e. maximum integer index + 1.
    embedding_dim: Integer. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings`
      matrix (see `keras.initializers`).
    embeddings_regularizer: Regularizer function applied to
      the `embeddings` matrix (see `keras.regularizers`).
    embeddings_constraint: Constraint function applied to
      the `embeddings` matrix (see `keras.constraints`).
    mask_zero: Boolean, whether or not the input value 0 is a special
      "padding" value that should be masked out. This is useful when using
      recurrent layers which may take variable length input. If this is
      `True`, then all subsequent layers in the model need to support masking
      or an exception will be raised. If mask_zero is set to True, as a
      consequence, index 0 cannot be used in the vocabulary (input_dim should
      equal size of vocabulary + 1).
    input_length: Length of input sequences, when it is constant.
      This argument is required if you are going to connect
      `Flatten` then `Dense` layers upstream
      (without it, the shape of the dense outputs cannot be computed).
    sparse: If True, calling this layer returns a `tf.SparseTensor`. If False,
      the layer returns a dense `tf.Tensor`. For an entry with no features in
      a sparse tensor (entry with value 0), the embedding vector of index 0 is
      returned by default.

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.

  **Note on variable placement:**
  By default, if a GPU is available, the embedding matrix will be placed on
  the GPU. This achieves the best performance, but it might cause issues:

  - You may be using an optimizer that does not support sparse GPU kernels.
  In this case you will see an error upon training your model.
  - Your embedding matrix may be too large to fit on your GPU. In this case
  you will see an Out Of Memory (OOM) error.

  In such cases, you should place the embedding matrix on the CPU memory.
  You can do so with a device scope, as such:

  ```python
  with tf.device('cpu:0'):
    embedding_layer = Embedding(...)
    embedding_layer.build()
  ```

  The pre-built `embedding_layer` instance can then be added to a `Sequential`
  model (e.g. `model.add(embedding_layer)`), called in a Functional model
  (e.g. `x = embedding_layer(x)`), or used in a subclassed model.
  """

  @utils.allow_initializer_layout
  def __init__(
    self,
    vocabulary_size,
    embedding_dim,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    sparse=False,
    **kwargs,
  ):
    if "input_shape" not in kwargs:
      if input_length:
        kwargs["input_shape"] = (input_length,)
      else:
        kwargs["input_shape"] = (None,)
    if vocabulary_size <= 0 or embedding_dim <= 0:
      raise ValueError(
        "Both `vocabulary_size` and `embedding_dim` should be positive, "
        f"Received vocabulary_size = {vocabulary_size} "
        f"and embedding_dim = {embedding_dim}"
      )
    if not base_layer_utils.v2_dtype_behavior_enabled() and "dtype" not in kwargs:
      # In TF1, the dtype defaults to the input dtype which is typically
      # int32, so explicitly set it to floatx
      kwargs["dtype"] = backend.floatx()
    # We set autocast to False, as we do not want to cast floating- point
    # inputs to self.dtype. In call(), we cast to int32, and casting to
    # self.dtype before casting to int32 might cause the int32 values to be
    # different due to a loss of precision.
    kwargs["autocast"] = False
    use_one_hot_matmul = kwargs.pop("use_one_hot_matmul", False)
    super().__init__(**kwargs)

    self.input_dim = vocabulary_size
    self.output_dim = embedding_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.embeddings_constraint = constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero
    self.supports_masking = mask_zero
    self.input_length = input_length
    self.sparse = sparse
    if self.sparse and self.mask_zero:
      raise ValueError(
        "`mask_zero` cannot be enabled when `tf.keras.layers.Embedding` is used with `tf.SparseTensor` input."
      )
    # Make this flag private and do not serialize it for now.
    # It will be part of the public API after further testing.
    self._use_one_hot_matmul = use_one_hot_matmul

  @tf_utils.shape_type_conversion
  def build(self, input_shape=None):
    self.embeddings = self.add_weight(
      shape=(self.input_dim, self.output_dim),
      initializer=self.embeddings_initializer,
      name="embeddings",
      regularizer=self.embeddings_regularizer,
      constraint=self.embeddings_constraint,
      experimental_autocast=False,
    )
    self.built = True

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return tf.not_equal(inputs, 0)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.input_length is None:
      return input_shape + (self.output_dim,)
    else:
      # input_length can be tuple if input is 3D or higher
      if isinstance(self.input_length, (list, tuple)):
        in_lens = list(self.input_length)
      else:
        in_lens = [self.input_length]
      if len(in_lens) != len(input_shape) - 1:
        raise ValueError(f'"input_length" is {self.input_length}, but received input has shape {input_shape}')
      else:
        for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
          if s1 is not None and s2 is not None and s1 != s2:
            raise ValueError(f'"input_length" is {self.input_length}, but received input has shape {input_shape}')
          elif s1 is None:
            in_lens[i] = s2
      return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != "int32" and dtype != "int64":
      inputs = tf.cast(inputs, "int32")
    if isinstance(inputs, tf.sparse.SparseTensor):
      if self.sparse:
        # get sparse embedding values
        embedding_values = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs.values)
        embedding_values = tf.reshape(embedding_values, [-1])
        # get sparse embedding indices
        indices_values_embed_axis = tf.range(self.output_dim)
        repeat_times = [inputs.indices.shape[0]]
        indices_values_embed_axis = tf.expand_dims(tf.tile(indices_values_embed_axis, repeat_times), -1)
        indices_values_embed_axis = tf.cast(indices_values_embed_axis, dtype=tf.int64)
        current_indices = tf.repeat(inputs.indices, [self.output_dim], axis=0)
        new_indices = tf.concat([current_indices, indices_values_embed_axis], 1)
        new_shape = tf.concat(
          [tf.cast(inputs.shape, dtype=tf.int64), [self.output_dim]],
          axis=-1,
        )
        out = tf.SparseTensor(
          indices=new_indices,
          values=embedding_values,
          dense_shape=new_shape,
        )
      else:
        sparse_inputs_expanded = tf.sparse.expand_dims(inputs, axis=-1)
        out = tf.nn.safe_embedding_lookup_sparse(
          embedding_weights=self.embeddings,
          sparse_ids=sparse_inputs_expanded,
          default_id=0,
        )
    elif self._use_one_hot_matmul:
      # Note that we change the dtype of the one_hot to be same as the
      # weight tensor, since the input data are usually ints, and weights
      # are floats. The nn.embedding_lookup support ids as ints, but
      # the one_hot matmul need both inputs and weights to be same dtype.
      one_hot_data = tf.one_hot(inputs, depth=self.input_dim, dtype=self.dtype)
      out = tf.matmul(one_hot_data, self.embeddings)
    else:
      out = tf.nn.embedding_lookup(self.embeddings, inputs)

    if self.sparse and not isinstance(out, tf.SparseTensor):
      out = tf.sparse.from_dense(out)

    if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
      # Instead of casting the variable as in most layers, cast the
      # output, as this is mathematically equivalent but is faster.
      out = tf.cast(out, self._dtype_policy.compute_dtype)
    return out

  def get_config(self):
    config = {
      "vocabulary_size": self.input_dim,
      "embedding_dim": self.output_dim,
      "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
      "embeddings_regularizer": regularizers.serialize(self.embeddings_regularizer),
      "activity_regularizer": regularizers.serialize(self.activity_regularizer),
      "embeddings_constraint": constraints.serialize(self.embeddings_constraint),
      "mask_zero": self.mask_zero,
      "input_length": self.input_length,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class EmbeddingGroup(tf.keras.layers.Layer):
  def __init__(self, feature_map: pd.DataFrame, trainable=True, **kwargs):
    super().__init__(trainable, **kwargs)
    self.feature_map = feature_map
    self.embedding_layers = {}
    self.hash_long_kernel = {}
    for name, dim, voc_size, hash_size in self.feature_map[(self.feature_map["ftype"] == "Categorical")][
      ["name", "dim", "voc_size", "hash_size"]
    ].values:
      if not math.isnan(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      self.embedding_layers[name] = Embedding(embedding_dim=dim, vocabulary_size=voc_size, name="embedding_" + name)

  def call(self, inputs: Dict[str, tf.Tensor], *args, **kwargs) -> Dict[str, tf.Tensor]:
    embedding_out = {}
    for name, hash_size in self.feature_map[(self.feature_map["ftype"] == "Categorical")][["name", "hash_size"]].values:
      input_tensor = inputs[name]
      if not math.isnan(hash_size):
        input_tensor = self.hash_long_kernel[name](input_tensor)
      input_tensor = self.embedding_layers[name](input_tensor)
      embedding_out[name] = input_tensor
    return embedding_out

  def save_checkpoint(self, checkpoint_path):
    for e in self.embedding_layers:
      e.save_checkpoint(checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    for e in self.embedding_layers:
      e.restore_checkpoint(checkpoint_path)


class JointEmbeddingInitializer(tf.keras.initializers.Initializer):
  def __init__(self, table_sizes, embedding_dim, wrapped):
    self.table_sizes = table_sizes
    self.wrapped = wrapped
    self.embedding_dim = embedding_dim

  def __call__(self, shape, dtype=tf.float32):
    with tf.device("/CPU:0"):
      subtables = []
      for table_size in self.table_sizes:
        subtable = self.wrapped()(shape=[table_size, self.embedding_dim], dtype=dtype)
        subtables.append(subtable)
      weights = tf.concat(subtables, axis=0)
    return weights

  def get_config(self):
    return {}


class EmbeddingInitializer(tf.keras.initializers.Initializer):
  def __call__(self, shape, dtype=tf.float32):
    with tf.device("/CPU:0"):
      maxval = tf.sqrt(tf.constant(1.0) / tf.cast(shape[0], tf.float32))
      maxval = tf.cast(maxval, dtype=dtype)
      minval = -maxval

      weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
      weights = tf.cast(weights, dtype=dtype)
    return weights

  def get_config(self):
    return {}


class JointEmbedding(tf.keras.layers.Layer):
  def __init__(self, table_sizes, output_dim, dtype, feature_names=None, trainable=True):
    super(JointEmbedding, self).__init__(dtype=dtype)
    self.table_sizes = table_sizes
    self.output_dim = output_dim
    self.embedding_table = None
    self.offsets = np.array([0] + table_sizes, dtype=np.int32).cumsum()
    self.offsets.reshape([1, -1])
    self.offsets = tf.constant(self.offsets, dtype=tf.int32)
    self.feature_names = feature_names
    if not self.feature_names:
      self.feature_names = ["feature_{i}" for i in range(len(table_sizes))]
    self.trainable = trainable

  def build(self, input_shape):
    initializer = JointEmbeddingInitializer(
      table_sizes=self.table_sizes, embedding_dim=self.output_dim, wrapped=EmbeddingInitializer
    )

    self.embedding_table = self.add_weight(
      "embedding_table",
      shape=[self.offsets[-1], self.output_dim],
      dtype=self.dtype,
      initializer=initializer,
      trainable=self.trainable,
    )

  def call(self, indices):
    indices = indices + self.offsets[:-1]
    return tf.nn.embedding_lookup(params=self.embedding_table, ids=indices)

  def save_checkpoint(self, checkpoint_path):
    for j in range(len(self.offsets) - 1):
      nrows = self.offsets[j + 1] - self.offsets[j]
      name = self.feature_names[j]
      filename = get_variable_path(checkpoint_path, name)

      indices = tf.range(start=self.offsets[j], limit=self.offsets[j] + nrows, dtype=tf.int32)
      arr = tf.gather(params=self.embedding_table, indices=indices, axis=0)
      arr = arr.numpy()
      np.save(arr=arr, file=filename)

  def restore_checkpoint(self, checkpoint_path):
    for j in range(len(self.offsets) - 1):
      name = self.feature_names[j]

      filename = get_variable_path(checkpoint_path, name)
      numpy_arr = np.load(file=filename)

      indices = tf.range(start=self.offsets[j], limit=self.offsets[j] + numpy_arr.shape[0], dtype=tf.int32)
      update = tf.IndexedSlices(values=numpy_arr, indices=indices, dense_shape=self.embedding_table.shape)
      self.embedding_table.scatter_update(sparse_delta=update)


class DualEmbeddingGroup(tf.keras.layers.Layer):
  """
  A group of embeddings with the same output dimension.
  If it runs out of GPU memory it will use CPU memory for the largest tables.
  """

  def __init__(
    self,
    cardinalities,
    output_dim,
    memory_threshold,
    cpu_embedding="multitable",
    gpu_embedding="fused",
    dtype=tf.float32,
    feature_names=None,
    trainable=True,
  ):
    # TODO: throw an exception if the features are not sorted by cardinality in reversed order

    super(DualEmbeddingGroup, self).__init__(dtype=dtype)

    if dtype not in [tf.float32, tf.float16]:
      raise ValueError(f"Only float32 and float16 embedding dtypes are currently supported. Got {dtype}.")

    cpu_embedding_class = EmbeddingGroup if cpu_embedding == "multitable" else JointEmbedding
    gpu_embedding_class = EmbeddingGroup if gpu_embedding == "multitable" else JointEmbedding

    cardinalities = np.array(cardinalities)

    self.memory_threshold = memory_threshold

    self.bytes_per_element = 2 if self.dtype == tf.float16 else 4

    self.table_sizes = cardinalities * output_dim * self.bytes_per_element
    self._find_first_gpu_index()
    self.cpu_cardinalities = cardinalities[: self.first_gpu_index]
    self.gpu_cardinalities = cardinalities[self.first_gpu_index :]

    if not feature_names:
      feature_names = [f"feature_{i}" for i in range(len(self.table_sizes))]

    self.feature_names = feature_names

    self.gpu_embedding = gpu_embedding_class(
      table_sizes=self.gpu_cardinalities.tolist(),
      output_dim=output_dim,
      dtype=self.dtype,
      feature_names=feature_names[self.first_gpu_index :],
      trainable=trainable,
    )

    # Force using FP32 for CPU embeddings, FP16 performance is much worse
    self.cpu_embeddings = cpu_embedding_class(
      table_sizes=self.cpu_cardinalities,
      output_dim=output_dim,
      dtype=tf.float32,
      feature_names=feature_names[: self.first_gpu_index],
      trainable=trainable,
    )

  def _find_first_gpu_index(self):
    # order from smallest to largest
    reversed_sizes = np.flip(self.table_sizes)
    cumulative_size = np.cumsum(reversed_sizes)
    cumulative_indicators = (cumulative_size > self.memory_threshold * 2**30).tolist()
    if True in cumulative_indicators:
      reversed_index = cumulative_indicators.index(True)
    else:
      reversed_index = len(cumulative_size)

    # convert to index into the original unreversed order
    index = len(reversed_sizes) - reversed_index
    self.first_gpu_index = index

  def call(self, indices):
    indices = tf.stack(indices, axis=1)

    to_concat = []
    if self.first_gpu_index > 0:
      # at least one cpu-based embedding
      cpu_indices = indices[:, : self.first_gpu_index]
      with tf.device("/CPU:0"):
        cpu_results = self.cpu_embeddings(cpu_indices)
        cpu_results = tf.cast(cpu_results, dtype=self.dtype)
        to_concat.append(cpu_results)

    if self.first_gpu_index < len(self.table_sizes):
      # at least one gpu-based embedding
      gpu_indices = indices[:, self.first_gpu_index :]
      gpu_results = self.gpu_embedding(gpu_indices)
      to_concat.append(gpu_results)

    if len(to_concat) > 1:
      result = tf.concat(to_concat, axis=1)
    else:
      result = to_concat[0]
    return result

  def save_checkpoint(self, checkpoint_path):
    self.gpu_embedding.save_checkpoint(checkpoint_path)
    self.cpu_embeddings.save_checkpoint(checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    self.gpu_embedding.restore_checkpoint(checkpoint_path)
    self.cpu_embeddings.restore_checkpoint(checkpoint_path)


class MaskedEmbeddingMean(tf.keras.layers.Layer):
  def __init__(self, hash_bucket_size, embedding_dim):
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(
      input_dim=hash_bucket_size, output_dim=embedding_dim, embeddings_initializer="glorot_uniform"
    )
    super(MaskedEmbeddingMean, self).__init__()

  def call(self, inputs):
    original_embedding = self.embedding(inputs)
    mask_tensor = 1 - tf.cast(inputs == 0, tf.float32)  # batch, len
    embedding_mask_tensor = tf.repeat(
      tf.expand_dims(mask_tensor, axis=-1), self.embedding_dim, axis=-1
    )  # batch, len, dim
    mean_tensor = tf.math.divide_no_nan(
      tf.reduce_sum(original_embedding * embedding_mask_tensor, axis=[1]), tf.reduce_sum(embedding_mask_tensor, axis=1)
    )
    return tf.expand_dims(mean_tensor, axis=1)


class QREmbedding(tf.keras.layers.Layer):
  """
  [Quotient-remainder](https://arxiv.org/abs/1909.02107) trick, by Hao-Jun Michael Shi et al., which reduces the number of embedding vectors to store, yet produces unique embedding vector for each item without explicit definition.

  The Quotient-Remainder technique works as follows. For a set of vocabulary and  embedding size
  `embedding_dim`, instead of creating a `vocabulary_size X embedding_dim` embedding table,
  we create *two* `num_buckets X embedding_dim` embedding tables, where `num_buckets`
  is much smaller than `vocabulary_size`.
  An embedding for a given item `index` is generated via the following steps:

  1. Compute the `quotient_index` as `index // num_buckets`.
  2. Compute the `remainder_index` as `index % num_buckets`.
  3. Lookup `quotient_embedding` from the first embedding table using `quotient_index`.
  4. Lookup `remainder_embedding` from the second embedding table using `remainder_index`.
  5. Return `quotient_embedding` * `remainder_embedding`.

  This technique not only reduces the number of embedding vectors needs to be stored and trained,
  but also generates a *unique* embedding vector for each item of size `embedding_dim`.
  Note that `q_embedding` and `r_embedding` can be combined using other operations,
  like `Add` and `Concatenate`.
  """

  def __init__(self, vocabulary, embedding_dim, num_buckets, name=None):
    super().__init__(name=name)
    self.num_buckets = num_buckets

    self.index_lookup = keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)
    self.q_embeddings = keras.layers.Embedding(
      num_buckets,
      embedding_dim,
    )
    self.r_embeddings = keras.layers.Embedding(
      num_buckets,
      embedding_dim,
    )

  def call(self, inputs):
    # Get the item index.
    embedding_index = self.index_lookup(inputs)
    # Get the quotient index.
    quotient_index = tf.math.floordiv(embedding_index, self.num_buckets)
    # Get the reminder index.
    remainder_index = tf.math.floormod(embedding_index, self.num_buckets)
    # Lookup the quotient_embedding using the quotient_index.
    quotient_embedding = self.q_embeddings(quotient_index)
    # Lookup the remainder_embedding using the remainder_index.
    remainder_embedding = self.r_embeddings(remainder_index)
    # Use multiplication as a combiner operation
    return quotient_embedding * remainder_embedding


class MDEmbedding(keras.layers.Layer):
  """
  [Mixed Dimension embeddings](https://arxiv.org/abs/1909.11810), by Antonio Ginart et al., which stores embedding vectors with mixed dimensions, where less popular items have reduced dimension embeddings.

  In the mixed dimension embedding technique, we train embedding vectors with full dimensions
  for the frequently queried items, while train embedding vectors with *reduced dimensions*
  for less frequent items, plus a *projection weights matrix* to bring low dimension embeddings
  to the full dimensions.

  More precisely, we define *blocks* of items of similar frequencies. For each block,
  a `block_vocab_size X block_embedding_dim` embedding table and `block_embedding_dim X full_embedding_dim`
  projection weights matrix are created. Note that, if `block_embedding_dim` equals `full_embedding_dim`,
  the projection weights matrix becomes an *identity* matrix. Embeddings for a given batch of item
  `indices` are generated via the following steps:

  1. For each block, lookup the `block_embedding_dim` embedding vectors using `indices`, and
  project them to the `full_embedding_dim`.
  2. If an item index does not belong to a given block, an out-of-vocabulary embedding is returned.
  Each block will return a `batch_size X full_embedding_dim` tensor.
  3. A mask is applied to the embeddings returned from each block in order to convert the
  out-of-vocabulary embeddings to vector of zeros. That is, for each item in the batch,
  a single non-zero embedding vector is returned from the all block embeddings.
  4. Embeddings retrieved from the blocks are combined using *sum* to produce the final
  `batch_size X full_embedding_dim` tensor.
  """

  def __init__(self, blocks_vocabulary, blocks_embedding_dims, base_embedding_dim, name=None):
    super().__init__(name=name)
    self.num_blocks = len(blocks_vocabulary)

    # Create vocab to block lookup.
    keys = []
    values = []
    for block_idx, block_vocab in enumerate(blocks_vocabulary):
      keys.extend(block_vocab)
      values.extend([block_idx] * len(block_vocab))
    self.vocab_to_block = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1)

    self.block_embedding_encoders = []
    self.block_embedding_projectors = []

    # Create block embedding encoders and projectors.
    for idx in range(self.num_blocks):
      vocabulary = blocks_vocabulary[idx]
      embedding_dim = blocks_embedding_dims[idx]
      block_embedding_encoder = self.embedding_encoder(vocabulary, embedding_dim, num_oov_indices=1)
      self.block_embedding_encoders.append(block_embedding_encoder)
      if embedding_dim == base_embedding_dim:
        self.block_embedding_projectors.append(keras.layers.Lambda(lambda x: x))
      else:
        self.block_embedding_projectors.append(keras.layers.Dense(units=base_embedding_dim))

    self.base_embedding_dim = 64

  def embedding_encoder(self, vocabulary, embedding_dim, num_oov_indices=0, name=None):
    return keras.Sequential(
      [
        keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=num_oov_indices),
        keras.layers.Embedding(input_dim=len(vocabulary) + num_oov_indices, output_dim=embedding_dim),
      ],
      name=f"{name}_embedding" if name else None,
    )

  def call(self, inputs):
    # Get block index for each input item.
    block_indicies = self.vocab_to_block.lookup(inputs)
    # Initialize output embeddings to zeros.
    embeddings = tf.zeros(shape=(tf.shape(inputs)[0], self.base_embedding_dim))
    # Generate embeddings from blocks.
    for idx in range(self.num_blocks):
      # Lookup embeddings from the current block.
      block_embeddings = self.block_embedding_encoders[idx](inputs)
      # Project embeddings to base_embedding_dim.
      block_embeddings = self.block_embedding_projectors[idx](block_embeddings)
      # Create a mask to filter out embeddings of items that do not belong to the current block.
      mask = tf.expand_dims(tf.cast(block_indicies == idx, tf.dtypes.float32), 1)
      # Set the embeddings for the items not belonging to the current block to zeros.
      block_embeddings = block_embeddings * mask
      # Add the block embeddings to the final embeddings.
      embeddings += block_embeddings

    return embeddings
