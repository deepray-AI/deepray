# -*- coding:utf-8 -*-
"""Dynamic Embedding layer."""

from collections import defaultdict
from typing import Dict, List
from typing import Optional, Literal

import pandas as pd
import tensorflow as tf
from absl import flags
from tensorflow.python.keras import regularizers, initializers

from deepray.layers.bucketize import NumericaBucketIdLayer, Hash
from deepray.utils import logging_util
from deepray.utils.horovod_utils import get_world_size, get_rank, is_main_process

logger = logging_util.get_logger()

try:
  import tensorflow_recommenders_addons as tfra
  from tensorflow_recommenders_addons import dynamic_embedding as de
  from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import BasicEmbedding as DynamicEmbedding
  from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import HvdAllToAllEmbedding

  class EmbeddingLayerRedis(DynamicEmbedding):
    def __init__(self, mini_batch_regularizer=None, mask_value=None, **kwargs):
      self.mini_batch_regularizer = regularizers.get(mini_batch_regularizer)
      self.mask_value = mask_value
      super().__init__(**kwargs)

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
        "mini_batch_regularizer": initializers.serialize(self.mini_batch_regularizer),
        "mask_value": self.mask_value,
      }
      base_config = super(EmbeddingLayerRedis, self).get_config()

      return dict(list(base_config.items()) + list(config.items()))

  class EmbeddingLayerGPU(DynamicEmbedding):
    def __init__(self, mini_batch_regularizer=None, mask_value=None, **kwargs):
      self.mini_batch_regularizer = regularizers.get(mini_batch_regularizer)
      self.mask_value = mask_value
      self.with_unique = kwargs.get("with_unique", True)
      super().__init__(**kwargs)

    def call(self, ids):
      with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
        if self.with_unique:
          ids_flat = tf.reshape(ids, [-1])
          unique_ids, idx = tf.unique(ids_flat)
          unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
          embeddings_flat = tf.gather(unique_embeddings, idx)
          embeddings_shape = tf.concat([tf.shape(ids), tf.constant(self.embedding_size, shape=(1,))], 0)
          embeddings = tf.reshape(embeddings_flat, embeddings_shape)
        else:
          embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, ids)
        return embeddings

    def get_config(self):
      config = {
        "mini_batch_regularizer": initializers.serialize(self.mini_batch_regularizer),
        "mask_value": self.mask_value,
      }
      base_config = super(EmbeddingLayerGPU, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

except ImportError as e:
  logger.warning("An exception occurred when import tensorflow_recommenders_addons: " + str(e))


class DistributedDynamicEmbedding(tf.keras.layers.Layer):
  def get_de_options(self, case, init_capacity, **kwargs):
    redis_creator = None
    cuckoo_creator = None
    hkv_creator = None

    if case == "Redis":
      if flags.FLAGS.redis_config_env:
        redis_config = tfra.dynamic_embedding.RedisTableConfig(redis_config_abs_dir_env=flags.FLAGS.redis_config_env)
      else:
        redis_config = tfra.dynamic_embedding.RedisTableConfig(redis_config_abs_dir=flags.FLAGS.redis_config_dir)
      redis_creator = tfra.dynamic_embedding.RedisTableCreator(redis_config)

    if case == "HKV":
      hkv_config = tfra.dynamic_embedding.HkvHashTableConfig(
        init_capacity=init_capacity,
        max_capacity=kwargs.get("max_capacity", 128 * 1024 * 1024),
        max_hbm_for_values=kwargs.get("max_hbm_for_values", 4 * 1024 * 1024 * 1024),
      )
      if flags.FLAGS.use_horovod:
        hkv_creator = tfra.dynamic_embedding.HkvHashTableCreator(
          hkv_config, saver=de.FileSystemSaver(proc_size=get_world_size(), proc_rank=get_rank())
        )
      else:
        hkv_creator = tfra.dynamic_embedding.HkvHashTableCreator(hkv_config, saver=de.FileSystemSaver())

    if flags.FLAGS.use_horovod:
      cuckoo_creator = de.CuckooHashTableCreator(
        saver=de.FileSystemSaver(proc_size=get_world_size(), proc_rank=get_rank())
      )
    else:
      cuckoo_creator = de.CuckooHashTableCreator(saver=de.FileSystemSaver())

    switcher = {
      "Redis": {
        "devices": ["/CPU:0"],
        "kv_creator": redis_creator,
      },
      "DRAM": {
        "devices": ["/CPU:0"],
        "kv_creator": cuckoo_creator,
      },
      "HBM": {
        "devices": ["/GPU:0"],
        "kv_creator": cuckoo_creator,
      },
      "HKV": {
        "devices": ["/GPU:0"],
        "kv_creator": hkv_creator,
      },
    }
    return switcher.get(case, None)

  def __init__(
    self,
    embedding_dim: int,
    key_dtype: str,
    value_dtype: str,
    initializer=None,
    name: str = "",
    device: Optional[Literal["HBM", "DRAM", "Redis", "HKV", "EV"]] = "DRAM",
    init_capacity=1 * 1024 * 1024,
    **kwargs,
  ):
    super(DistributedDynamicEmbedding, self).__init__()
    self.embedding_dim = embedding_dim
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.initializer = initializer
    self.device = device
    self.init_capacity = init_capacity

    if device == "Redis":
      de_option = self.get_de_options(device, init_capacity, **kwargs)
      self.emb = EmbeddingLayerRedis(
        embedding_size=embedding_dim,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        initializer=initializer,
        name=name,
        devices=de_option["devices"],
        kv_creator=de_option["kv_creator"],
        **kwargs,
      )
      if is_main_process():
        logger.info(f"Create EmbeddingLayer for {name} on {device} with {embedding_dim} dim")
      return

    de_option = self.get_de_options(device, init_capacity, **kwargs)
    if not flags.FLAGS.use_horovod:
      self.emb = EmbeddingLayerGPU(
        embedding_size=embedding_dim,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        initializer=initializer,
        name=name,
        devices=de_option["devices"],
        init_capacity=init_capacity,
        kv_creator=de_option["kv_creator"],
        **kwargs,
      )
      if is_main_process():
        logger.info(f"Create EmbeddingLayer for {name} on {device} with {embedding_dim} dim")
    else:
      self.emb = HvdAllToAllEmbedding(
        embedding_size=embedding_dim,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        initializer=initializer,
        name=name,
        devices=de_option["devices"],
        init_capacity=init_capacity,
        kv_creator=de_option["kv_creator"],
        **kwargs,
      )
      if is_main_process():
        logger.info(f"Create HvdAllToAllEmbedding for {name} on {device} with {embedding_dim} dim")

  def call(self, ids, *args, **kwargs):
    return self.emb(ids)

  def get_config(self):
    config = super().get_config()
    config.update({
      "embedding_dim": self.embedding_dim,
      "key_dtype": self.key_dtype,
      "value_dtype": self.value_dtype,
      "initializer": self.initializer,
      "device": self.device,
      "init_capacity": self.init_capacity,
    })
    return config


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
    composition_size: int,
    complementary_strategy: str = "Q-R",
    operation: str = "add",
    name: str = "",
    device: Optional[Literal["HBM", "DRAM", "Redis", "HKV"]] = None,
    initializer=None,
    **kwargs,
  ):
    super(CompositionalEmbedding, self).__init__()
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
    self.composition_factor = self.factor2decimal(composition_size)
    self.complementary_strategy = complementary_strategy
    self.operation = operation
    self.suffix = name
    self.initializer = initializer

  def factor2decimal(self, composition_part: int):
    if self.key_dtype == "int32":
      base = 32
    elif self.key_dtype == "int64":
      base = 64
    else:
      raise ValueError(f"{self.key_dtype} type not support yet.")

    # Calculate the quotient and remainder of A divided by composition_size.
    quotient = base // composition_part
    remainder = base % composition_part

    # Create a list of length composition_size with each element equal to the quotient.
    result = [quotient] * composition_part

    # Distribute the remainder among the first few elements of the list.
    for i in range(remainder):
      result[i] += 1

    # Sort the list in ascending order.
    result.sort()

    res = []
    for i in range(len(result)):
      binary_str = ""
      for j in range(len(result)):
        binary_str += result[j] * ("1" if i == j else "0")

      int_num = int(binary_str, 2) - 2**base if int(binary_str[0]) else int(binary_str, 2)
      res.append(int_num)
    return res

  def build(self, input_shape=None):
    self.composition_emb = EmbeddingVariable(
      embedding_dim=self.embedding_dim,
      key_dtype=self.key_dtype,
      value_dtype=self.value_dtype,
      initializer=self.initializer,
      name=f"embeddings_{self.suffix}/Compositional",
      device=self.device,
    )

  def call(self, inputs, *args, **kwargs):
    ids_QRP = [tf.bitwise.bitwise_and(inputs, x) for x in self.composition_factor]
    results = tf.split(self.composition_emb(ids_QRP), num_or_size_splits=len(ids_QRP), axis=0)
    new_result = [tf.squeeze(x, axis=0) for x in results]
    if self.operation == "add":
      ret = tf.add_n(new_result)
      return ret
    elif self.operation == "mul":
      ret = tf.multiply(new_result)
      return ret
    elif self.operation == "concat":
      ret = tf.concat(new_result, 1)
      return ret
    else:
      raise ValueError(f"{self.operation} operation not support yet.")


class DiamondEmbedding(tf.keras.layers.Layer):
  """
  Diamond Brother(金刚葫芦娃) has all the powers of the seven brothers, so should the Diamond Embedding too.
  """

  def __init__(self, feature_map: pd.DataFrame, fold_columns: Dict[str, List[str]], **kwargs):
    super(DiamondEmbedding, self).__init__()
    columns = ["bucket_boundaries", "hash_size", "voc_size", "composition_size", "storage_type"]
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
        dim_value = df.loc[df["name"] == name]["dim"].values[0]
        dim_values.append(dim_value)
        folder_map[name] = key
      if len(set(dim_values)) != 1:
        raise ValueError(
          f"Cannot aggregate {group} because dimensions are not equal. Names: {group}, Dims: {dim_values}"
        )

    # Record the remaining features that do not need to be folded
    for name in self.feature_map[~(self.feature_map["ftype"].isin(["Label", "Weight"]))]["name"].values:
      if name not in folder_map:
        folder_map[name] = name
    return folder_map

  def build(self, input_shape):
    self.embedding_layers = {}
    self.hash_long_kernel = {}
    self.numerical_bucket_kernel = {}
    self.split_dims = defaultdict(list)
    for (
      name,
      length,
      dim,
      voc_size,
      dtype,
      hash_size,
      composition_factor,
      storage_type,
      bucket_boundaries,
    ) in self.feature_map[~(self.feature_map["ftype"].isin(["Label", "Weight"]))][
      [
        "name",
        "length",
        "dim",
        "voc_size",
        "dtype",
        "hash_size",
        "composition_size",
        "storage_type",
        "bucket_boundaries",
      ]
    ].values:
      if self.is_valid_value(bucket_boundaries):
        bucket_boundaries_list = sorted(set(map(float, bucket_boundaries.split(","))))
        self.numerical_bucket_kernel[name] = NumericaBucketIdLayer(bucket_boundaries_list)

      if self.is_valid_value(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      if self.fold_columns[name] not in self.embedding_layers:
        composition_factor = (
          self.feature_map.loc[self.feature_map["name"] == self.fold_columns[name]]["composition_size"].values[0]
          if self.fold_columns[name] in self.feature_map["name"].values
          else composition_factor
        )
        storage_type = (
          self.feature_map.loc[self.feature_map["name"] == self.fold_columns[name]]["storage_type"].values[0]
          if self.fold_columns[name] in self.feature_map["name"].values
          else storage_type
        )
        if self.is_valid_value(composition_factor):
          self.embedding_layers[self.fold_columns[name]] = CompositionalEmbedding(
            embedding_dim=dim,
            key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
            value_dtype=tf.float32,
            composition_size=composition_factor,
            operation="add",
            name=self.fold_columns[name],
          )
        else:
          self.embedding_layers[self.fold_columns[name]] = EmbeddingVariable(
            embedding_dim=dim,
            key_dtype=tf.int32 if self.is_valid_value(bucket_boundaries) else dtype,
            value_dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform(),
            name="embedding_" + self.fold_columns[name],
            device=storage_type,
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
      ~(self.feature_map["ftype"].isin(["Label", "Weight"]))
    ][["code", "name", "hash_size", "bucket_boundaries"]].values:
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
