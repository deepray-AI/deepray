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
        kv_creator=kwargs.get("kv_creator") if kwargs.get("kv_creator", None) else de_option["kv_creator"],
        **kwargs,
      )
      if is_main_process():
        logger.info(f"Create EmbeddingLayer for {name} on {device} with {embedding_dim} dim")
      return

    de_option = self.get_de_options(device, init_capacity, **kwargs)
    if kwargs.get("kv_creator", None):
      kv_creator = kwargs.pop("kv_creator")
    else:
      kv_creator = de_option["kv_creator"]
    if not flags.FLAGS.use_horovod:
      self.emb = EmbeddingLayerGPU(
        embedding_size=embedding_dim,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        initializer=initializer,
        name=name,
        devices=de_option["devices"],
        init_capacity=init_capacity,
        kv_creator=kv_creator,
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
        kv_creator=kv_creator,
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
