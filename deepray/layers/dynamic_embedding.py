# -*- coding:utf-8 -*-
"""Dynamic Embedding layer."""

from collections import defaultdict
from typing import Dict, List
from typing import Optional, Literal

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
from absl import flags, logging
from keras import initializers
from keras import regularizers
from tensorflow.python.keras import regularizers, initializers
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import BasicEmbedding as DynamicEmbedding
from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import HvdAllToAllEmbedding

from deepray.layers.bucketize import NumericaBucketIdLayer, Hash

FLAGS = flags.FLAGS


class EmbeddingLayerRedis(DynamicEmbedding):

  def __init__(self, mini_batch_regularizer=None, devices=['/CPU:0'], mask_value=None, **kwargs):
    self.mini_batch_regularizer = regularizers.get(mini_batch_regularizer)
    self.mask_value = mask_value

    if FLAGS.redis_config_env:
      redis_config = tfra.dynamic_embedding.RedisTableConfig(redis_config_abs_dir_env=FLAGS.redis_config_env)
    else:
      redis_config = tfra.dynamic_embedding.RedisTableConfig(redis_config_abs_dir=FLAGS.redis_config_dir)

    super(EmbeddingLayerRedis,
          self).__init__(devices=devices, kv_creator=tfra.dynamic_embedding.RedisTableCreator(redis_config), **kwargs)

  def call(self, ids):
    with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
      ids_flat = tf.reshape(ids, [-1])
      with tf.device("/CPU:0"):
        unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat([tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
      return embeddings

  def get_config(self):
    config = {
        'mini_batch_regularizer': initializers.serialize(self.mini_batch_regularizer),
        'mask_value': self.mask_value
    }
    base_config = super(EmbeddingLayerRedis, self).get_config()

    return dict(list(base_config.items()) + list(config.items()))

  def compute_mask(self, inputs, mask=None):
    if not self.mask_value:
      return None
    return tf.not_equal(inputs, self.mask_value)


class EmbeddingLayerGPU(DynamicEmbedding):

  def call(self, ids):
    with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
      ids_flat = tf.reshape(ids, [-1])
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat([tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
      return embeddings


class DistributedDynamicEmbedding():

  def __init__(
      self,
      embedding_dim: int,
      key_dtype: str,
      value_dtype: str,
      initializer=None,
      name: str = '',
      device: Optional[Literal["HBM", "DRAM", "Redis"]] = None,
  ):
    if device == "Redis":
      self.emb = EmbeddingLayerRedis(
          embedding_size=embedding_dim,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=initializer,
          name=name
      )
      logging.info(f"Create EmbeddingLayerRedis for {name} on {device} with {embedding_dim} dim")
      return
    elif device == "HBM":
      self.devices = ['/GPU:0']
    elif device == "DRAM":
      self.devices = ['/CPU:0']
    else:
      self.devices = ['/GPU:0']

    if not FLAGS.use_horovod:
      self.emb = EmbeddingLayerGPU(
          embedding_size=embedding_dim,
          devices=self.devices,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=initializer,
          name=name,
          init_capacity=1000000 * 8,  # 如果提示hash冲突，调整该参数
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver())
      )
      logging.info(f"Create EmbeddingLayerGPU for {name} on {device} with {embedding_dim} dim")
    else:
      import horovod.tensorflow as hvd

      mpi_size = hvd.size()
      mpi_rank = hvd.rank()
      self.emb = HvdAllToAllEmbedding(
          # mpi_size=mpi_size,
          embedding_size=embedding_dim,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=initializer,
          init_capacity=800000,
          name=name,
          devices=self.devices,
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))
      )
      logging.info(f"Create HvdAllToAllEmbedding for {name} on {device} with {embedding_dim} dim")

  def __call__(self, ids, *args, **kwargs):
    return self.emb(ids)


class CompositionalEmbedding(tf.keras.layers.Layer):
  """
  Compositional Embedding is designed for reducing Large-scale Sparse Embedding Weights.
  See [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems]
  (https://arxiv.org/abs/1909.02107)
  """

  def __init__(
      self,
      embedding_dim: int,
      key_dtype: str,
      value_dtype: str,
      composition_factor: str = "",
      complementary_strategy: str = "Q-R",
      operation: str = "add",
      name: str = '',
      device: Optional[Literal["HBM", "DRAM", "Redis"]] = None,
      initializer=None,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.device = device
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
    self.value_dtype = value_dtype
    self.composition_factor = self.factor2decimal(composition_factor) if composition_factor else self.factor2decimal(
        "16,16"
    ) if self.key_dtype == "int32" else self.factor2decimal("32,32")
    self.complementary_strategy = complementary_strategy
    self.operation = operation
    self.suffix = name
    self.initializer = initializer

  def factor2decimal(self, composition_factor: str):
    """
    print(binary_to_decimal("16,16"))  # 4294901760, 65535
    print(binary_to_decimal("17,15"))  # 4294934528, 32767
    print(binary_to_decimal("15,17"))  # 4294836224, 131071
    print(binary_to_decimal("14,18"))  # 4294705152, 262143
    print(binary_to_decimal("13,19"))  # 4294443008, 524287

    print(binary_to_decimal("32,32"))  # 18446744069414584320, 4294967295
    print(binary_to_decimal("33,31"))  # 18446744071562067968, 2147483647
    print(binary_to_decimal("31,33"))  # 18446744065119617024, 8589934591
    print(binary_to_decimal("30,34"))  # 18446744056529682432, 17179869183
    print(binary_to_decimal("29,35"))  # 18446744039349813248, 34359738367
    """
    if self.key_dtype == "int32":
      Q, R = map(int, composition_factor.split(','))
      Q = (1 << Q) - 1 << (32 - Q)
      R = (1 << R) - 1
      return Q, R
    elif self.key_dtype == "int64":
      """
  sign bit
      1   111111111111111111111 111111111111111111111 111111111111111111111     -9223372036854776000
      1   000000000000000000000 111111111111111111111 111111111111111111111     -9223367638808264705
      1   111111111111111111111 000000000000000000000 111111111111111111111     -4398044413953
      1   111111111111111111111 111111111111111111111 000000000000000000000     -2097151
      """
      Q, R, P = -9223367638808264705, -4398044413953, -2097152
      return tf.constant(Q, dtype=tf.int64), tf.constant(R, dtype=tf.int64), tf.constant(P, dtype=tf.int64)
    else:
      raise ValueError(f"{self.key_dtype} type not support yet.")

  def build(self, input_shape=None):
    self.composition_emb = DistributedDynamicEmbedding(
        embedding_dim=self.embedding_dim,
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        initializer=self.initializer,
        name=f"embeddings_{self.suffix}/Compositional",
        device=self.device,
    )

  def call(self, inputs, *args, **kwargs):
    if self.key_dtype == "int32":
      ids_Q = tf.bitwise.bitwise_and(inputs, self.composition_factor[0])
      ids_R = tf.bitwise.bitwise_and(inputs, self.composition_factor[1])
      result_Q, result_R = tf.split(self.composition_emb([ids_Q, ids_R]), num_or_size_splits=2, axis=0)
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
    elif self.key_dtype == "int64":
      ids_Q = tf.bitwise.bitwise_and(inputs, self.composition_factor[0])
      ids_R = tf.bitwise.bitwise_and(inputs, self.composition_factor[1])
      ids_P = tf.bitwise.bitwise_and(inputs, self.composition_factor[2])
      result_Q, result_R, result_P = tf.split(self.composition_emb([ids_Q, ids_R, ids_P]), num_or_size_splits=3, axis=0)
      result_Q = tf.squeeze(result_Q, axis=0)
      result_R = tf.squeeze(result_R, axis=0)
      result_P = tf.squeeze(result_P, axis=0)
      if self.operation == "add":
        ret = tf.add_n([result_Q, result_R, result_P])
        return ret
      if self.operation == "mul":
        ret = tf.multiply(result_Q, result_R, result_P)
        return ret
      if self.operation == "concat":
        ret = tf.concat([result_Q, result_R, result_P], 1)
        return ret
    else:
      raise ValueError(f"{self.key_dtype} type not support yet.")


class DiamondEmbedding(tf.keras.layers.Layer):
  """
  Diamond Brother(金刚葫芦娃) has all the powers of the seven brothers, so should the Diamond Embedding too.
  """

  def __init__(self, feature_map: pd.DataFrame, fold_columns: Dict[str, List[str]], **kwargs):
    super().__init__(**kwargs)
    columns = ["bucket_boundaries", "hash_size", "voc_size", "composition_factor", "storage_type"]
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
        raise ValueError(
            f"Cannot aggregate {group} because dimensions are not equal. Names: {group}, Dims: {dim_values}"
        )

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
    for name, length, dim, voc_size, dtype, hash_size, composition_factor, storage_type, bucket_boundaries in self.feature_map[
        ~(self.feature_map['ftype'].isin(["Label", "Weight"]))][[
            "name", "length", "dim", "voc_size", "dtype", "hash_size", "composition_factor", "storage_type",
            "bucket_boundaries"
        ]].values:

      if self.is_valid_value(bucket_boundaries):
        bucket_boundaries_list = sorted(set(map(float, bucket_boundaries.split(","))))
        self.numerical_bucket_kernel[name] = NumericaBucketIdLayer(bucket_boundaries_list)

      if self.is_valid_value(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      if self.fold_columns[name] not in self.embedding_layers:
        composition_factor = self.feature_map.loc[self.feature_map['name'] == self.fold_columns[name]
                                                 ]['composition_factor'].values[0] if self.fold_columns[
                                                     name] in self.feature_map['name'].values else composition_factor
        storage_type = self.feature_map.loc[
            self.feature_map['name'] == self.fold_columns[name]
        ]['storage_type'].values[0] if self.fold_columns[name] in self.feature_map['name'].values else storage_type
        if self.is_valid_value(composition_factor):
          self.embedding_layers[self.fold_columns[name]] = CompositionalEmbedding(
              embedding_dim=dim,
              key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
              composition_factor=composition_factor,
              operation="add",
              name=self.fold_columns[name]
          )
        else:
          if storage_type == "Redis":
            redis_config = tfra.dynamic_embedding.RedisTableConfig(redis_config_abs_dir=FLAGS.config_file)
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
              self.embedding_layers[self.fold_columns[name]] = DynamicEmbedding(
                  embedding_size=dim,
                  key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                  value_dtype=tf.float32,
                  initializer=tf.keras.initializers.GlorotUniform(),
                  name='embedding_' + self.fold_columns[name],
                  init_capacity=800000,
                  devices=devices,
                  kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver())
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
                      saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank)
                  )
              )

          else:
            devices = ['/GPU:0']
            if not FLAGS.use_horovod:
              self.embedding_layers[self.fold_columns[name]] = DynamicEmbedding(
                  embedding_size=dim,
                  key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
                  value_dtype=tf.float32,
                  initializer=tf.keras.initializers.GlorotUniform(),
                  name='embedding_' + self.fold_columns[name],
                  init_capacity=800000,
                  devices=devices,
                  kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver())
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
                      saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank)
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

  def call(self, inputs, *args, **kwargs) -> Dict[str, List[tf.Tensor]]:
    result = defaultdict(list)
    id_tensors = defaultdict(list)
    for code, name, hash_size, bucket_boundaries in self.feature_map[
        ~(self.feature_map['ftype'].isin(["Label", "Weight"]))][["code", "name", "hash_size",
                                                                 "bucket_boundaries"]].values:

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
      embedding_out = tf.split(embedding_out_concat, num_or_size_splits=self.split_dims_final[fold_name], axis=1)

      for i, embedding in enumerate(embedding_out):  # LENGTH(embedding_out) == split_dims_final
        if self.is_sequence_feature[fold_name][i] and True:
          embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)  # (feature_combin_num, (batch, x, 16*6+1))
        result[fold_name].append(embedding)

    return result
