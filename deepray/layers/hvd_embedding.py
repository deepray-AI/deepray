import tensorflow as tf

from .embedding import DynamicEmbedding
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import make_partition
from tensorflow_recommenders_addons import dynamic_embedding as de


class HvdAllToAllEmbedding(DynamicEmbedding):
  """
  This embedding layer will dispatch keys to all corresponding Horovod workers and receive its own keys for distributed training before embedding_lookup.
  """

  def __init__(self,
               with_unique=True,
               mpi_size=None,
               batch_size=None,
               *args,
               **kwargs):
    try:
      import horovod.tensorflow as hvd
    except ImportError:
      raise ValueError(
        "Please install Horovod first if you want to use distributed synchronous training based on Horovod"
      )
    self.hvd = hvd
    self.with_unique = with_unique
    self.batch_size = batch_size
    if mpi_size is None:
      self._mpi_size = self.hvd.size()
    else:
      self._mpi_size = mpi_size
    super(HvdAllToAllEmbedding, self).__init__(*args, **kwargs)

  def __relocate_dense_feature__(self, ids, batch_size=None):
    """
    Args:
      ids: A 2-D Tensor with shape: (batch_size, sequence_length) or a 1-D Tensor with shape: (batch_size,).
        If batch_size is provided, then it trust the batch_size argument, to avoid new an OP instead.
      batch_size: Integer or a int32/int64 scalar. All ranks must have same batch_size.
        Otherwise will make undefined behavior.

    Returns:
      flat_reloc_ids: a flat ids partitioned to each rank.
    """
    if ids.dtype not in (tf.int32, tf.int64):
      raise NotImplementedError

    if ids.shape.rank > 2:
      raise NotImplementedError(
        'Input ids must be shape '
        f'(batch_size, sequence_length) or (batch_size,), but get {ids.shape}'
      )

    if batch_size is None:
      input_shape = tf.shape(ids)
      batch_size = input_shape[0]

    partition_index = self.params.partition_fn(ids, self._mpi_size)
    ids_partitions, ids_indices = make_partition(ids, partition_index,
                                                 self._mpi_size)
    partitions_sizes = tf.stack([tf.size(p) for p in ids_partitions], axis=0)
    relocs_tensor = tf.concat(ids_partitions, axis=0)
    flat_reloc_ids, remote_sizes = self.hvd.alltoall(relocs_tensor,
                                                     splits=partitions_sizes)
    return flat_reloc_ids, remote_sizes, ids_indices

  def __alltoall_embedding_lookup__(self, ids):
    if self._mpi_size == 1:
      return de.shadow_ops.embedding_lookup(self.shadow, ids)
    if isinstance(ids, tf.sparse.SparseTensor):
      raise NotImplementedError('SparseTensor is not supported yet.')

    input_shape = tf.shape(ids)
    if self.batch_size is None:
      batch_size_runtime = input_shape[0]

    reloc_ids, remote_sizes, gather_indices = self.__relocate_dense_feature__(
      ids, batch_size=batch_size_runtime)

    lookup_result = de.shadow_ops.embedding_lookup(self.shadow, reloc_ids)
    lookup_result, _ = self.hvd.alltoall(lookup_result, splits=remote_sizes)

    recover_shape = tf.concat((input_shape, (self.embedding_size,)), axis=0)
    gather_indices = tf.expand_dims(tf.concat(gather_indices, axis=0), axis=-1)
    lookup_result = tf.scatter_nd(gather_indices, lookup_result, recover_shape)
    return lookup_result

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
    ids_flat = tf.reshape(ids, (-1,))
    if self.with_unique:
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = self.__alltoall_embedding_lookup__(unique_ids)
      lookup_result = tf.gather(unique_embeddings, idx)
    else:
      lookup_result = self.__alltoall_embedding_lookup__(ids_flat)
    lookup_result = tf.reshape(
      lookup_result, tf.concat([input_shape, [self.embedding_size]], 0))
    return lookup_result

  def get_config(self):
    config = super(HvdAllToAllEmbedding, self).get_config()
    config.update({"with_unique": self.with_unique})
    config.update({"mpi_size": self._mpi_size})
    config.update({"batch_size": self.batch_size})
    return config
