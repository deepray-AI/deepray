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
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
from absl import flags
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine import base_layer_utils
from keras.engine.base_layer import Layer
from keras.utils import tf_utils
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import dynamic_embedding_variable as devar

from deepray.layers.bucketize import NumericaBucketIdLayer, Hash

FLAGS = flags.FLAGS


def get_variable_path(checkpoint_path, name, i=0):
  tokens = name.split('/')
  tokens = [t for t in tokens if 'model_parallel' not in t and 'data_parallel' not in t]
  name = '_'.join(tokens)
  name = name.replace(':', '_')
  filename = name + f'_part{i}' + '.npy'
  return os.path.join(checkpoint_path, filename)


# @keras_export("keras.layers.Embedding")
class Embedding(Layer):
  """Turns positive integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on positive integer inputs of a fixed range. The
    `tf.keras.layers.TextVectorization`, `tf.keras.layers.StringLookup`,
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
      3D tensor with shape: `(batch_size, input_length, embedding_dim)`.

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
      embedding_dim,
      vocabulary_size=None,
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
    if (
        not base_layer_utils.v2_dtype_behavior_enabled()
        and "dtype" not in kwargs
    ):
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

    self.vocabulary_size = vocabulary_size
    self.embedding_dim = embedding_dim
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
        "`mask_zero` cannot be enabled when "
        "`tf.keras.layers.Embedding` is used with `tf.SparseTensor` "
        "input."
      )
    # Make this flag private and do not serialize it for now.
    # It will be part of the public API after further testing.
    self._use_one_hot_matmul = use_one_hot_matmul

  @tf_utils.shape_type_conversion
  def build(self, input_shape=None):
    self.embeddings = self.add_weight(
      shape=(self.vocabulary_size, self.embedding_dim),
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
      return input_shape + (self.embedding_dim,)
    else:
      # input_length can be tuple if input is 3D or higher
      if isinstance(self.input_length, (list, tuple)):
        in_lens = list(self.input_length)
      else:
        in_lens = [self.input_length]
      if len(in_lens) != len(input_shape) - 1:
        raise ValueError(
          f'"input_length" is {self.input_length}, but received '
          f"input has shape {input_shape}"
        )
      else:
        for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
          if s1 is not None and s2 is not None and s1 != s2:
            raise ValueError(
              f'"input_length" is {self.input_length}, but '
              f"received input has shape {input_shape}"
            )
          elif s1 is None:
            in_lens[i] = s2
      return (input_shape[0],) + tuple(in_lens) + (self.embedding_dim,)

  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != "int32" and dtype != "int64":
      inputs = tf.cast(inputs, "int32")
    if isinstance(inputs, tf.sparse.SparseTensor):
      if self.sparse:
        # get sparse embedding values
        embedding_values = tf.nn.embedding_lookup(
          params=self.embeddings, ids=inputs.values
        )
        embedding_values = tf.reshape(embedding_values, [-1])
        # get sparse embedding indices
        indices_values_embed_axis = tf.range(self.embedding_dim)
        repeat_times = [inputs.indices.shape[0]]
        indices_values_embed_axis = tf.expand_dims(
          tf.tile(indices_values_embed_axis, repeat_times), -1
        )
        indices_values_embed_axis = tf.cast(
          indices_values_embed_axis, dtype=tf.int64
        )
        current_indices = tf.repeat(
          inputs.indices, [self.embedding_dim], axis=0
        )
        new_indices = tf.concat(
          [current_indices, indices_values_embed_axis], 1
        )
        new_shape = tf.concat(
          [tf.cast(inputs.shape, dtype=tf.int64), [self.embedding_dim]],
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
      one_hot_data = tf.one_hot(
        inputs, depth=self.vocabulary_size, dtype=self.dtype
      )
      out = tf.matmul(one_hot_data, self.embeddings)
    else:
      out = tf.nn.embedding_lookup(self.embeddings, inputs)

    if self.sparse and not isinstance(out, tf.SparseTensor):
      out = tf.sparse.from_dense(out)

    if (
        self._dtype_policy.compute_dtype
        != self._dtype_policy.variable_dtype
    ):
      # Instead of casting the variable as in most layers, cast the
      # output, as this is mathematically equivalent but is faster.
      out = tf.cast(out, self._dtype_policy.compute_dtype)
    return out

  def get_config(self):
    config = {
      "vocabulary_size": self.vocabulary_size,
      "embedding_dim": self.embedding_dim,
      "embeddings_initializer": initializers.serialize(
        self.embeddings_initializer
      ),
      "embeddings_regularizer": regularizers.serialize(
        self.embeddings_regularizer
      ),
      "activity_regularizer": regularizers.serialize(
        self.activity_regularizer
      ),
      "embeddings_constraint": constraints.serialize(
        self.embeddings_constraint
      ),
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
    for name, dim, voc_size, hash_size in self.feature_map[(self.feature_map['ftype'] == "Categorical")][["name", "dim", "voc_size", "hash_size"]].values:
      if not math.isnan(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      self.embedding_layers[name] = Embedding(
        embedding_dim=dim,
        vocabulary_size=voc_size,
        name='embedding_' + name)

  def call(self, inputs: Dict[str, tf.Tensor], *args, **kwargs) -> Dict[str, tf.Tensor]:
    embedding_out = {}
    for name, hash_size in self.feature_map[(self.feature_map['ftype'] == "Categorical")][["name", "hash_size"]].values:
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
    with tf.device('/CPU:0'):
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
    with tf.device('/CPU:0'):
      maxval = tf.sqrt(tf.constant(1.) / tf.cast(shape[0], tf.float32))
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
      self.feature_names = ['feature_{i}' for i in range(len(table_sizes))]
    self.trainable = trainable

  def build(self, input_shape):
    initializer = JointEmbeddingInitializer(table_sizes=self.table_sizes,
                                            embedding_dim=self.output_dim,
                                            wrapped=EmbeddingInitializer)

    self.embedding_table = self.add_weight("embedding_table",
                                           shape=[self.offsets[-1], self.output_dim],
                                           dtype=self.dtype,
                                           initializer=initializer,
                                           trainable=self.trainable)

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

  def __init__(self, cardinalities, output_dim, memory_threshold,
               cpu_embedding='multitable', gpu_embedding='fused', dtype=tf.float32,
               feature_names=None, trainable=True):

    # TODO: throw an exception if the features are not sorted by cardinality in reversed order

    super(DualEmbeddingGroup, self).__init__(dtype=dtype)

    if dtype not in [tf.float32, tf.float16]:
      raise ValueError(f'Only float32 and float16 embedding dtypes are currently supported. Got {dtype}.')

    cpu_embedding_class = EmbeddingGroup if cpu_embedding == 'multitable' else JointEmbedding
    gpu_embedding_class = EmbeddingGroup if gpu_embedding == 'multitable' else JointEmbedding

    cardinalities = np.array(cardinalities)

    self.memory_threshold = memory_threshold

    self.bytes_per_element = 2 if self.dtype == tf.float16 else 4

    self.table_sizes = cardinalities * output_dim * self.bytes_per_element
    self._find_first_gpu_index()
    self.cpu_cardinalities = cardinalities[:self.first_gpu_index]
    self.gpu_cardinalities = cardinalities[self.first_gpu_index:]

    if not feature_names:
      feature_names = [f'feature_{i}' for i in range(len(self.table_sizes))]

    self.feature_names = feature_names

    self.gpu_embedding = gpu_embedding_class(table_sizes=self.gpu_cardinalities.tolist(),
                                             output_dim=output_dim, dtype=self.dtype,
                                             feature_names=feature_names[self.first_gpu_index:],
                                             trainable=trainable)

    # Force using FP32 for CPU embeddings, FP16 performance is much worse
    self.cpu_embeddings = cpu_embedding_class(table_sizes=self.cpu_cardinalities,
                                              output_dim=output_dim, dtype=tf.float32,
                                              feature_names=feature_names[:self.first_gpu_index],
                                              trainable=trainable)

  def _find_first_gpu_index(self):
    # order from smallest to largest
    reversed_sizes = np.flip(self.table_sizes)
    cumulative_size = np.cumsum(reversed_sizes)
    cumulative_indicators = (cumulative_size > self.memory_threshold * 2 ** 30).tolist()
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
      cpu_indices = indices[:, :self.first_gpu_index]
      with tf.device('/CPU:0'):
        cpu_results = self.cpu_embeddings(cpu_indices)
        cpu_results = tf.cast(cpu_results, dtype=self.dtype)
        to_concat.append(cpu_results)

    if self.first_gpu_index < len(self.table_sizes):
      # at least one gpu-based embedding
      gpu_indices = indices[:, self.first_gpu_index:]
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


class DynamicEmbedding(tf.keras.layers.Layer):
  """
  A keras style Embedding layer. The `Embedding` layer acts same like
  [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding),
  except that the `Embedding` has dynamic embedding space so it does
  not need to set a static vocabulary size, and there will be no hash conflicts
  between features.

  The embedding layer allow arbirary input shape of feature ids, and get
  (shape(ids) + embedding_size) lookup result. Normally the first dimension
  is batch_size.

  ### Example
  ```python
  embedding = dynamic_embedding.keras.layers.Embedding(8) # embedding size 8
  ids = tf.constant([[15,2], [4,92], [22,4]], dtype=tf.int64) # (3, 2)
  out = embedding(ids) # lookup result, (3, 2, 8)
  ```

  You could inherit the `Embedding` class to implement a custom embedding
  layer with other fixed shape output.

  TODO(Lifann) Currently the Embedding only implemented in eager mode
  API, need to support graph mode also.
  """

  def __init__(self,
               embedding_size,
               key_dtype=tf.int64,
               value_dtype=tf.float32,
               combiner='sum',
               initializer=None,
               devices=None,
               name='DynamicEmbeddingLayer',
               with_unique=True,
               **kwargs):
    """
    Creates a Embedding layer.

    Args:
      embedding_size: An object convertible to int. Length of embedding vector
        to every feature id.
      key_dtype: Dtype of the embedding keys to weights. Default is int64.
      value_dtype: Dtype of the embedding weight values. Default is float32
      combiner: A string or a function to combine the lookup result. It's value
        could be 'sum', 'mean', 'min', 'max', 'prod', 'std', etc. whose are
        one of tf.math.reduce_xxx.
      initializer: Initializer to the embedding values. Default is RandomNormal.
      devices: List of devices to place the embedding layer parameter.
      name: Name of the embedding layer.
      with_unique: Bool. Whether if the layer does unique on `ids`. Default is True.

      **kwargs:
        trainable: Bool. Whether if the layer is trainable. Default is True.
        bp_v2: Bool. If true, the embedding layer will be updated by incremental
          amount. Otherwise, it will be updated by value directly. Default is
          False.
        restrict_policy: A RestrictPolicy class to restrict the size of
          embedding layer parameter since the dynamic embedding supports
          nearly infinite embedding space capacity.
        init_capacity: Integer. Initial number of kv-pairs in an embedding
          layer. The capacity will growth if the used space exceeded current
          capacity.
        partitioner: A function to route the keys to specific devices for
          distributed embedding parameter.
        kv_creator: A KVCreator object to create external KV storage as
          embedding parameter.
        max_norm: If not `None`, each values is clipped if its l2-norm is larger
        distribute_strategy: Used when creating ShadowVariable.
        keep_distribution: Bool. If true, save and restore python object with
          devices information. Default is false.
    """

    try:
      embedding_size = int(embedding_size)
    except:
      raise TypeError(
        'embeddnig size must be convertible to integer, but get {}'.format(
          type(embedding_size)))

    self.embedding_size = embedding_size
    self.combiner = combiner
    if initializer is None:
      initializer = tf.keras.initializers.RandomNormal()
    partitioner = kwargs.get('partitioner', devar.default_partition_fn)
    trainable = kwargs.get('trainable', True)
    self.max_norm = kwargs.get('max_norm', None)
    self.keep_distribution = kwargs.get('keep_distribution', False)
    self.with_unique = with_unique

    parameter_name = name + '-parameter' if name else 'EmbeddingParameter'
    with tf.name_scope('DynamicEmbedding'):
      self.params = de.get_variable(parameter_name,
                                    key_dtype=key_dtype,
                                    value_dtype=value_dtype,
                                    dim=self.embedding_size,
                                    devices=devices,
                                    partitioner=partitioner,
                                    shared_name='layer_embedding_variable',
                                    initializer=initializer,
                                    trainable=trainable,
                                    checkpoint=kwargs.get('checkpoint', True),
                                    init_size=kwargs.get('init_capacity', 0),
                                    kv_creator=kwargs.get('kv_creator', None),
                                    restrict_policy=kwargs.get(
                                      'restrict_policy', None),
                                    bp_v2=kwargs.get('bp_v2', False))

      self.distribute_strategy = kwargs.get('distribute_strategy', None)
      shadow_name = name + '-shadow' if name else 'ShadowVariable'
      self.shadow = de.shadow_ops.ShadowVariable(
        self.params,
        name=shadow_name,
        max_norm=self.max_norm,
        trainable=trainable,
        distribute_strategy=self.distribute_strategy)
    self._current_ids = self.shadow.ids
    self._current_exists = self.shadow.exists
    self.optimizer_vars = self.shadow._optimizer_vars
    super(DynamicEmbedding, self).__init__(name=name,
                                           trainable=trainable,
                                           dtype=value_dtype)

  def call(self, ids):
    """
    Compute embedding output for feature ids. The output shape will be (shape(ids),
    embedding_size).

    Args:
      ids: feature ids of the input. It should be same dtype as the key_dtype
        of the layer.

    Returns:
      A embedding output with shape (shape(ids), embedding_size).
    """
    ids = tf.convert_to_tensor(ids)
    input_shape = tf.shape(ids)
    embeddings_shape = tf.concat([input_shape, [self.embedding_size]], 0)
    ids_flat = tf.reshape(ids, (-1,))
    if self.with_unique:
      with tf.name_scope(self.name + "/EmbeddingWithUnique"):
        unique_ids, idx = tf.unique(ids_flat)
        unique_embeddings = de.shadow_ops.embedding_lookup(
          self.shadow, unique_ids)
        embeddings_flat = tf.gather(unique_embeddings, idx)
    else:
      embeddings_flat = de.shadow_ops.embedding_lookup(self.shadow, ids_flat)
    embeddings = tf.reshape(embeddings_flat, embeddings_shape)
    return embeddings

  def get_config(self):
    _initializer = self.params.initializer
    if _initializer is None:
      _initializer = tf.keras.initializers.Zeros()
    _max_norm = None
    if isinstance(self.max_norm, tf.keras.constraints.Constraint):
      _max_norm = tf.keras.constraints.serialize(self.max_norm)

    if self.params.restrict_policy:
      _restrict_policy = self.params.restrict_policy.__class__
    else:
      _restrict_policy = None

    config = {
      'embedding_size':
        self.embedding_size,
      'key_dtype':
        self.params.key_dtype,
      'value_dtype':
        self.params.value_dtype,
      'combiner':
        self.combiner,
      'initializer':
        tf.keras.initializers.serialize(_initializer),
      'devices':
        self.params.devices if self.keep_distribution else None,
      'name':
        self.name,
      'trainable':
        self.trainable,
      'bp_v2':
        self.params.bp_v2,
      'restrict_policy':
        _restrict_policy,
      'init_capacity':
        self.params.init_size,
      'partitioner':
        self.params.partition_fn,
      'kv_creator':
        self.params.kv_creator if self.keep_distribution else None,
      'max_norm':
        _max_norm,
      'distribute_strategy':
        self.distribute_strategy,
    }
    return config


class EmbeddingLayerGPU(DynamicEmbedding):
  def call(self, ids):
    with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
      ids_flat = tf.reshape(ids, [-1])
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat(
        [tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
      return embeddings


class EmbeddingLayerRedis(DynamicEmbedding):
  def call(self, ids):
    with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
      ids_flat = tf.reshape(ids, [-1])
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat([tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
      return embeddings


class MultiHashEmbedding(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, key_dtype, multihash_factor="16,16", complementary_strategy="Q-R", operation="add", name: str = '',
               **kwargs):
    super().__init__(**kwargs)
    strategy_list = ["Q-R"]
    op_list = ["add", "mul", "concat"]

    if complementary_strategy not in strategy_list:
      raise ValueError("The strategy %s is not supported" % complementary_strategy)
    if operation not in op_list:
      raise ValueError("The operation %s is not supported" % operation)
    # if complementary_strategy == 'Q-R':
    #   if num_of_partitions != 2:
    #     raise ValueError("the num_of_partitions must be 2 when using Q-R strategy.")

    self.embedding_dim = embedding_dim
    self.key_dtype = key_dtype
    self.multihash_factor = self.factor2decimal(multihash_factor)
    self.complementary_strategy = complementary_strategy
    self.operation = operation
    self.suffix = name

  def factor2decimal(self, multihash_factor: str):
    """
    print(binary_to_decimal("16,16"))  # 4294901760, 65535
    print(binary_to_decimal("17,15"))  # 4294934528, 32767
    print(binary_to_decimal("15,17"))  # 4294836224, 131071
    print(binary_to_decimal("14,18"))  # 4294705152, 262143
    print(binary_to_decimal("13,19"))  # 4294443008, 524287
    """
    if self.key_dtype == "int32":
      Q, R = map(int, multihash_factor.split(','))
      Q = (1 << Q) - 1 << (32 - Q)
      R = (1 << R) - 1
      return Q, R

  def build(self, input_shape=None):
    if not FLAGS.use_horovod:
      self.multihash_emb = EmbeddingLayerGPU(embedding_size=self.embedding_dim,
                                             key_dtype=self.key_dtype,
                                             value_dtype=tf.float32,
                                             initializer=tf.keras.initializers.GlorotUniform(),
                                             name=f"embeddings_{self.suffix}/multihash",
                                             init_capacity=800000,
                                             kv_creator=de.CuckooHashTableCreator(
                                               saver=de.FileSystemSaver())
                                             )

    else:
      import horovod.tensorflow as hvd

      gpu_device = ["GPU:0"]
      mpi_size = hvd.size()
      mpi_rank = hvd.rank()
      self.multihash_emb = de.keras.layers.HvdAllToAllEmbedding(embedding_size=self.embedding_dim,
                                                                key_dtype=self.key_dtype,
                                                                value_dtype=tf.float32,
                                                                initializer=tf.keras.initializers.GlorotUniform(),
                                                                devices=gpu_device,
                                                                init_capacity=800000,
                                                                name=f"embeddings_{self.suffix}/multihash",
                                                                kv_creator=de.CuckooHashTableCreator(
                                                                  saver=de.FileSystemSaver(proc_size=mpi_size,
                                                                                           proc_rank=mpi_rank))
                                                                )

  def call(self, inputs, *args, **kwargs):
    if self.complementary_strategy == "Q-R":
      ids_Q = tf.bitwise.bitwise_and(inputs, self.multihash_factor[0])
      ids_R = tf.bitwise.bitwise_and(inputs, self.multihash_factor[1])
      result_Q, result_R = tf.split(self.multihash_emb([ids_Q, ids_R]), num_or_size_splits=2, axis=0)
      result_Q = tf.squeeze(result_Q, axis=0)
      result_R = tf.squeeze(result_R, axis=0)
      if self.operation == "add":
        ret = tf.add(result_Q, result_R)
        return ret
      if self.operation == "mul":
        ret = tf.multiply(result_Q, result_R)
        return ret
      if self.operation == "concat":
        ret = tf.concat([result_Q, result_R], 1)
        return ret


class MaskedEmbeddingMean(tf.keras.layers.Layer):
  def __init__(self, hash_bucket_size, embedding_dim):
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(
      input_dim=hash_bucket_size,
      output_dim=embedding_dim,
      embeddings_initializer='glorot_uniform'
    )
    super(MaskedEmbeddingMean, self).__init__()

  def call(self, inputs):
    original_embedding = self.embedding(inputs)
    mask_tensor = 1 - tf.cast(inputs == 0, tf.float32)  # batch, len
    embedding_mask_tensor = tf.repeat(tf.expand_dims(mask_tensor, axis=-1), self.embedding_dim, axis=-1)  # batch, len, dim
    mean_tensor = tf.math.divide_no_nan(tf.reduce_sum(original_embedding * embedding_mask_tensor, axis=[1]), tf.reduce_sum(embedding_mask_tensor, axis=1))
    return tf.expand_dims(mean_tensor, axis=1)


class DiamondEmbedding(tf.keras.layers.Layer):
  """
  Diamond Brother(金刚葫芦娃) has all the powers of the seven brothers, so should the Diamond Embedding too.
  """

  def __init__(self, feature_map: pd.DataFrame, fold_columns: Dict[str, List[str]], **kwargs):
    super().__init__(**kwargs)
    columns = ["bucket_boundaries", "hash_size", "voc_size", "multihash_factor", "storage_type"]
    for col in columns:
      if col not in feature_map.columns:
        feature_map[col] = None

    self.feature_map = feature_map
    self.fold_columns = self.aggregate_by_dim(feature_map, fold_columns)

  def aggregate_by_dim(self, df: pd.DataFrame, fold_columns: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Aggregate the dim values for each group of names in the fold_columns list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        fold_columns (Dict[str, List[str]]): A list of lists of names to aggregate.

    Returns:
        Dict[str, str]: A dictionary containing the results of the aggregation.
    """
    folder_map = {}
    for key, group in fold_columns.items():
      dim_values = []
      for name in group:
        dim_value = df.loc[df['name'] == name]['dim'].values[0]
        dim_values.append(dim_value)
        folder_map[name] = key
      if len(set(dim_values)) != 1:
        raise ValueError(f"Cannot aggregate {group} because dimensions are not equal. Names: {group}, Dims: {dim_values}")

    # Record the remaining features that do not need to be folded
    for name in self.feature_map[~(self.feature_map['ftype'].isin(["Label", "Weight"]))]["name"].values:
      if name not in folder_map:
        folder_map[name] = name
    return folder_map

  def build(self, input_shape):
    self.embedding_layers = {}
    self.hash_long_kernel = {}
    self.numerical_bucket_kernel = {}
    self.split_dims = defaultdict(list)
    for name, length, dim, voc_size, dtype, hash_size, multihash_factor, storage_type, bucket_boundaries in self.feature_map[~(self.feature_map['ftype'].isin(["Label", "Weight"]))][
      ["name", "length", "dim", "voc_size", "dtype", "hash_size", "multihash_factor", "storage_type", "bucket_boundaries"]].values:

      if self.is_valid_value(bucket_boundaries):
        bucket_boundaries_list = sorted(set(map(float, bucket_boundaries.split(","))))
        self.numerical_bucket_kernel[name] = NumericaBucketIdLayer(bucket_boundaries_list)

      if self.is_valid_value(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      if self.fold_columns[name] not in self.embedding_layers:
        multihash_factor = self.feature_map.loc[self.feature_map['name'] == self.fold_columns[name]]['multihash_factor'].values[0] if self.fold_columns[name] in self.feature_map[
          'name'].values else multihash_factor
        storage_type = self.feature_map.loc[self.feature_map['name'] == self.fold_columns[name]]['storage_type'].values[0] if self.fold_columns[name] in self.feature_map[
          'name'].values else storage_type
        if self.is_valid_value(multihash_factor):
          self.embedding_layers[self.fold_columns[name]] = MultiHashEmbedding(
            embedding_dim=dim,
            key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
            multihash_factor=multihash_factor,
            operation="add",
            name=self.fold_columns[name]
          )
        else:
          if storage_type == "Redis":
            redis_config = tfra.dynamic_embedding.RedisTableConfig(
              redis_config_abs_dir=FLAGS.config_file
            )
            self.embedding_layers[self.fold_columns[name]] = EmbeddingLayerRedis(
              embedding_size=dim,
              key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
              value_dtype=tf.float32,
              initializer=tf.keras.initializers.GlorotUniform(),
              name='embedding_redis' + self.fold_columns[name],
              devices="/job:localhost/replica:0/task:0/CPU:0",
              kv_creator=de.RedisTableCreator(redis_config)
            )
          elif storage_type == "DRAM":
            devices = ['/CPU:0']
            if not FLAGS.use_horovod:
              self.embedding_layers[self.fold_columns[name]] = EmbeddingLayerGPU(
                embedding_size=dim,
                key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                value_dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                name='embedding_' + self.fold_columns[name],
                init_capacity=800000,
                devices=devices,
                kv_creator=de.CuckooHashTableCreator(
                  saver=de.FileSystemSaver()
                )
              )
            else:
              import horovod.tensorflow as hvd

              mpi_size = hvd.size()
              mpi_rank = hvd.rank()
              self.embedding_layers[self.fold_columns[name]] = de.keras.layers.HvdAllToAllEmbedding(
                # mpi_size=mpi_size,
                embedding_size=dim,
                key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                value_dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                init_capacity=800000,
                name='embedding_' + self.fold_columns[name],
                devices=devices,
                kv_creator=de.CuckooHashTableCreator(
                  saver=de.FileSystemSaver(proc_size=mpi_size,
                                           proc_rank=mpi_rank)
                )
              )

          else:
            devices = ['/GPU:0']
            if not FLAGS.use_horovod:
              self.embedding_layers[self.fold_columns[name]] = EmbeddingLayerGPU(
                embedding_size=dim,
                key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                value_dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                name='embedding_' + self.fold_columns[name],
                init_capacity=800000,
                devices=devices,
                kv_creator=de.CuckooHashTableCreator(
                  saver=de.FileSystemSaver()
                )
              )
            else:
              import horovod.tensorflow as hvd

              mpi_size = hvd.size()
              mpi_rank = hvd.rank()
              self.embedding_layers[self.fold_columns[name]] = de.keras.layers.HvdAllToAllEmbedding(
                # mpi_size=mpi_size,
                embedding_size=dim,
                key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                value_dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                init_capacity=800000,
                name='embedding_' + self.fold_columns[name],
                devices=devices,
                kv_creator=de.CuckooHashTableCreator(
                  saver=de.FileSystemSaver(proc_size=mpi_size,
                                           proc_rank=mpi_rank)
                )
              )

      self.split_dims[self.fold_columns[name]].append(length)
    # [1,1,1,10,1,1,30,1] -> [[3, 10, 2, 30, 1] and [False, True, False, True, False] for split sequence feature
    self.split_dims_final = defaultdict(list)
    self.is_sequence_feature = defaultdict(list)
    tmp_sum = defaultdict(int)
    for fold_name, dims in self.split_dims.items():
      for dim in dims:
        if dim == 1:
          tmp_sum[fold_name] += 1
        else:
          if tmp_sum[fold_name] > 0:
            self.split_dims_final[fold_name].append(tmp_sum[fold_name])
            self.is_sequence_feature[fold_name].append(False)
          self.split_dims_final[fold_name].append(dim)
          self.is_sequence_feature[fold_name].append(True)
          tmp_sum[fold_name] = 0

    for fold_name, _sum in tmp_sum.items():
      if _sum > 0:
        self.split_dims_final[fold_name].append(_sum)
        self.is_sequence_feature[fold_name].append(False)

  def is_valid_value(self, x):
    """
    x1 = '-2.87,-1.93,-1.32,-0.84,-0.42,-0.01,0.4,0.86,1.52'
    x2 = None
    x3 = np.nan
    x4 = ""
    x5 = 6

    print(is_valid_value(x1)) #  True
    print(is_valid_value(x2)) #  False
    print(is_valid_value(x3)) #  False
    print(is_valid_value(x4)) #  False
    print(is_valid_value(x5)) #  True
    """
    return isinstance(x, (int, str)) and bool(x)

  def call(self, inputs, *args, **kwargs):
    result = defaultdict(list)
    id_tensors = defaultdict(list)
    for code, name, hash_size, bucket_boundaries in self.feature_map[~(self.feature_map['ftype'].isin(["Label", "Weight"]))][
      ["code", "name", "hash_size", "bucket_boundaries"]].values:
      input_tensor = inputs[name]
      id_tensor_prefix_code = code << 47

      if self.is_valid_value(bucket_boundaries):
        input_tensor = self.numerical_bucket_kernel[name](input_tensor)

      if self.is_valid_value(hash_size):
        input_tensor = self.hash_long_kernel[name](input_tensor)

      id_tensor = tf.bitwise.bitwise_or(input_tensor, id_tensor_prefix_code)
      id_tensors[self.fold_columns[name]].append(id_tensor)

    for fold_name, id_tensor in id_tensors.items():
      id_tensors_concat = tf.concat(id_tensor, axis=1)
      embedding_out_concat = self.embedding_layers[fold_name](id_tensors_concat)
      embedding_out = tf.split(embedding_out_concat,
                               num_or_size_splits=self.split_dims_final[fold_name], axis=1)

      for i, embedding in enumerate(embedding_out):  # LENGTH(embedding_out) == split_dims_final
        if self.is_sequence_feature[fold_name][i] and True:
          embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)  # (feature_combin_num, (batch, x, 16*6+1))
        result[fold_name].append(embedding)

    return result
