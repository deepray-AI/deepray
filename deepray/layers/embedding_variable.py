# -*- coding:utf-8 -*-
"""Dynamic Embedding layer."""

import typing

import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from deepray.custom_ops.embedding_variable import config_pb2
from deepray.custom_ops.embedding_variable import variables as ev_variables
from deepray.custom_ops.embedding_variable.variable_scope import get_embedding_variable
from deepray.utils import logging_util
from deepray.utils.horovod_utils import get_world_size

logger = logging_util.get_logger()

StorageType = {
  "HBM": config_pb2.StorageType.HBM,
  "DRAM": config_pb2.StorageType.DRAM,
  "HBM_DRAM": config_pb2.StorageType.HBM_DRAM,
  "LEVELDB": config_pb2.StorageType.LEVELDB,
  "SSDHASH": config_pb2.StorageType.SSDHASH,
  "DRAM_LEVELDB": config_pb2.StorageType.DRAM_LEVELDB,
  "DRAM_SSDHASH": config_pb2.StorageType.DRAM_SSDHASH,
}

CacheStrategy = {"LFU": config_pb2.CacheStrategy.LFU, "LRU": config_pb2.CacheStrategy.LRU}


def default_partition_fn(keys, shard_num):
  """The default partition function.
    partition keys by "mod" strategy.

    keys: a tensor presents the keys to be partitioned.
    shard_num: the num of partitions
  Returns:
    a tensor with same shape as keys with type of `tf.int32`,
      represents the corresponding partition-ids of keys.
  """
  return math_ops.mod(keys, shard_num)


def int64_partition_fn(keys, shard_num):
  return math_ops.cast(math_ops.mod(keys, shard_num), dtype=dtypes.int32)


def partition_fn_v2(keys, shard_num):
  return tf.cast(
    tf.strings.to_hash_bucket_fast(
      tf.strings.as_string(keys),  # 将 int 转为 string 再哈希
      num_buckets=shard_num,
    ),
    tf.int32,
  )


class EmbeddingVariable(tf.keras.layers.Layer):
  def __init__(
    self,
    embedding_dim: int,
    key_dtype=dtypes.int64,
    value_dtype: str = None,
    initializer=None,
    name: str = "",
    with_unique=False,
    partition_fn: typing.Callable[[typing.Any, typing.Any], typing.Any] = None,
    **kwargs,
  ):
    super(EmbeddingVariable, self).__init__(name=name)
    self.embedding_size = embedding_dim
    self.with_unique = with_unique
    self.world_size = get_world_size()

    if partition_fn is None:
      if key_dtype == dtypes.int64:
        partition_fn = int64_partition_fn
      elif key_dtype == dtypes.int32:
        partition_fn = default_partition_fn

    storage_type = kwargs.get("storage_type", None)
    if storage_type:
      ev_option = ev_variables.EmbeddingVariableOption(
        storage_option=ev_variables.StorageOption(
          storage_type=StorageType[storage_type],
          storage_path=kwargs.get("storage_path", None),
          storage_size=kwargs.get("storage_size", [1024 * 1024 * 1024]),
          cache_strategy=CacheStrategy[kwargs.get("cache_strategy", "LFU")],
        )
      )
    else:
      ev_option = ev_variables.EmbeddingVariableOption()

    self.embedding_variable = get_embedding_variable(
      embedding_dim=embedding_dim,
      key_dtype=key_dtype,
      value_dtype=value_dtype,
      initializer=initializer,
      name=name,
      ev_option=ev_option,
    )

    self.partition_fn = partition_fn
    if self.world_size > 1:
      self.call = self.hvd_read
      if self.world_size >= 8:  # 小规模并行时用取模更快
        self.partition_fn = partition_fn_v2
    else:
      self.call = self.unique_read if self.with_unique else self.read

  def make_partition(self, data, partition_index):
    """
    Shard keys to shard_num partitions

    Args:
      data: keys or values, usually the IDs of dynamic features.
      partition_index: partitions index.
      shard_num: partition number
    Returns:
      a pair of tensor: (partition result, partition indices)
    """
    partitions = tf.dynamic_partition(data, partition_index, self.world_size)
    indices = tf.dynamic_partition(math_ops.range(array_ops.shape(data)[0]), partition_index, self.world_size)
    return partitions, indices

  def read(self, ids, *args, **kwargs):
    return self.embedding_variable.sparse_read(ids)

  def unique_read(self, ids, *args, **kwargs):
    """Read with deduplication for better performance with repeated IDs."""
    with ops.name_scope(f"{self.name}/EmbeddingWithUnique"):
      ids_flat = tf.reshape(ids, [-1])
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = self.embedding_variable.sparse_read(unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat([tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
    return embeddings

  def hvd_read(self, ids, *args, **kwargs):
    """
    Compute embedding output for feature ids. The output shape will be (shape(ids),
    embedding_size).

    Args:
      ids: feature ids of the input. It should be same dtype as the key_dtype
        of the layer.

    Returns:
      A embedding output with shape (shape(ids), embedding_size).
    """
    is_ragged = isinstance(ids, tf.RaggedTensor)

    if is_ragged:
      original_structure = ids
      ids = ids.flat_values

    input_shape = tf.shape(ids)
    embeddings_shape = tf.concat([input_shape, [self.embedding_size]], 0)

    ids_flat = tf.reshape(ids, [-1])

    def distributed_lookup(ids):
      partition_index = self.partition_fn(ids, self.world_size)
      ids_partitions, gather_indices = self.make_partition(ids, partition_index)
      partitions_sizes = tf.stack([tf.size(p) for p in ids_partitions], axis=0)
      relocs_tensor = tf.concat(ids_partitions, axis=0)
      # Provide a unique name for the first alltoall operation
      flat_reloc_ids, remote_sizes = hvd.alltoall(
        relocs_tensor, splits=partitions_sizes, name=f"{self.name}_alltoall_ids"
      )

      lookup_result = self.read(flat_reloc_ids)
      lookup_result, _ = hvd.alltoall(lookup_result, splits=remote_sizes, name=f"{self.name}_alltoall_embeddings")

      input_shape = tf.shape(ids)
      recover_shape = tf.concat((input_shape, (self.embedding_size,)), axis=0)
      gather_indices = tf.expand_dims(tf.concat(gather_indices, axis=0), axis=-1)
      lookup_result = tf.scatter_nd(gather_indices, lookup_result, recover_shape)
      return lookup_result

    if self.with_unique:
      # with ops.name_scope(name, "EmbeddingWithUnique"):
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = distributed_lookup(unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
    else:
      embeddings_flat = distributed_lookup(ids_flat)

    embeddings = tf.reshape(embeddings_flat, embeddings_shape)

    if is_ragged:
      embeddings = tf.RaggedTensor.from_row_lengths(embeddings, original_structure.row_lengths())

    return embeddings

  def get_config(self):
    config = super().get_config()
    config.update({
      "world_size": self.world_size,
      "name": self.name,
    })
    return config
