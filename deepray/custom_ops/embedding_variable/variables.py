from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import tf_export

from deepray.custom_ops.embedding_variable import config_pb2
from deepray.utils import logging_util

logger = logging_util.get_logger()


@tf_export(v1=["InitializerOption"])
class InitializerOption(object):
  def __init__(self, initializer=None, default_value_dim=4096, default_value_no_permission=0.0):
    self.initializer = initializer
    self.default_value_dim = default_value_dim
    self.default_value_no_permission = default_value_no_permission
    if default_value_dim <= 0:
      print("default value dim must larger than 1, the default value dim is set to default 4096.")
      default_value_dim = 4096


@tf_export(v1=["GlobalStepEvict"])
class GlobalStepEvict(object):
  def __init__(self, steps_to_live=None):
    self.steps_to_live = steps_to_live


@tf_export(v1=["L2WeightEvict"])
class L2WeightEvict(object):
  def __init__(self, l2_weight_threshold=-1.0):
    self.l2_weight_threshold = l2_weight_threshold
    if l2_weight_threshold <= 0 and l2_weight_threshold != -1.0:
      logger.warning("l2_weight_threshold is invalid, l2_weight-based eviction is disabled")


@tf_export(v1=["CheckpointOption"])
class CheckpointOption(object):
  def __init__(
    self, ckpt_to_load_from=None, tensor_name_in_ckpt=None, always_load_from_specific_ckpt=False, init_data_source=None
  ):
    self.ckpt_to_load_from = ckpt_to_load_from
    self.tensor_name_in_ckpt = tensor_name_in_ckpt
    self.always_load_from_specific_ckpt = always_load_from_specific_ckpt
    self.init_data_source = init_data_source


@tf_export(v1=["StorageOption"])
class StorageOption(object):
  def __init__(
    self,
    storage_type=None,
    storage_path=None,
    storage_size=[1024 * 1024 * 1024],
    cache_strategy=config_pb2.CacheStrategy.LFU,
    layout=None,
  ):
    self.storage_type = storage_type
    self.storage_path = storage_path
    self.storage_size = storage_size
    self.cache_strategy = cache_strategy
    self.layout = layout
    if not isinstance(storage_size, list):
      raise ValueError("storage_size should be list type")
    if len(storage_size) < 4:
      for i in range(len(storage_size), 4):
        storage_size.append(1024 * 1024 * 1024)
    if storage_path is not None:
      if storage_type is None:
        raise ValueError("storage_type musnt'be None when storage_path is set")
      else:
        if not file_io.file_exists(storage_path):
          file_io.recursive_create_dir(storage_path)
    else:
      if storage_type is not None and storage_type in [
        config_pb2.StorageType.LEVELDB,
        config_pb2.StorageType.SSDHASH,
        config_pb2.StorageType.DRAM_SSDHASH,
        config_pb2.StorageType.DRAM_LEVELDB,
      ]:
        raise ValueError("storage_path musnt'be None when storage_type is set")


@tf_export(v1=["EmbeddingVariableOption"])
class EmbeddingVariableOption(object):
  def __init__(
    self,
    ht_type="",
    ht_partition_num=1000,
    evict_option=None,
    ckpt=None,
    filter_option=None,
    storage_option=StorageOption(),
    init_option=InitializerOption(),
  ):
    self.ht_type = ht_type
    self.ht_partition_num = ht_partition_num
    self.evict = evict_option
    self.ckpt = ckpt
    self.filter_strategy = filter_option
    self.storage_option = storage_option
    self.init = init_option


@tf_export(v1=["CounterFilter"])
class CounterFilter(object):
  def __init__(self, filter_freq=0):
    self.filter_freq = filter_freq


@tf_export(v1=["CBFFilter"])
class CBFFilter(object):
  def __init__(self, filter_freq=0, max_element_size=0, false_positive_probability=-1.0, counter_type=dtypes.uint64):
    if false_positive_probability != -1.0:
      if false_positive_probability <= 0.0:
        raise ValueError("false_positive_probablity must larger than 0")
      else:
        if max_element_size <= 0:
          raise ValueError("max_element_size must larger than 0 when false_positive_probability is not -1.0")
    else:
      if max_element_size != 0:
        raise ValueError("max_element_size can't be set when false_probability is -1.0")
    self.max_element_size = max_element_size
    self.false_positive_probability = false_positive_probability
    self.counter_type = counter_type
    self.filter_freq = filter_freq


class EmbeddingVariableConfig(object):
  def __init__(
    self,
    steps_to_live=None,
    steps_to_live_l2reg=None,
    l2reg_theta=None,
    l2reg_lambda=None,
    l2_weight_threshold=-1.0,
    ht_type=None,
    filter_strategy=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    always_load_from_specific_ckpt=False,
    init_data_source=None,
    handle_name=None,
    emb_index=None,
    slot_index=None,
    block_num=None,
    primary=None,
    slot_num=None,
    storage_type=config_pb2.StorageType.DRAM,
    storage_path=None,
    storage_size=None,
    storage_cache_strategy=config_pb2.CacheStrategy.LFU,
    layout=None,
    default_value_dim=4096,
    default_value_no_permission=0.0,
  ):
    self.steps_to_live = steps_to_live
    self.steps_to_live_l2reg = steps_to_live_l2reg
    self.l2reg_theta = l2reg_theta
    self.l2reg_lambda = l2reg_lambda
    self.ckpt_to_load_from = ckpt_to_load_from
    self.tensor_name_in_ckpt = tensor_name_in_ckpt
    self.always_load_from_specific_ckpt = always_load_from_specific_ckpt
    self.init_data_source = init_data_source
    self.handle_name = handle_name
    self.emb_index = emb_index
    self.slot_index = slot_index
    self.block_num = block_num
    self.primary = primary
    self.slot_num = slot_num
    self.ht_type = ht_type
    self.l2_weight_threshold = l2_weight_threshold
    self.filter_strategy = filter_strategy
    self.storage_type = storage_type
    self.storage_path = storage_path
    self.storage_size = storage_size
    self.storage_cache_strategy = storage_cache_strategy
    self.layout = layout
    self.default_value_dim = default_value_dim
    self.default_value_no_permission = default_value_no_permission

  def reveal(self):
    if self.steps_to_live is None:
      self.steps_to_live = 0
    if self.steps_to_live_l2reg is None:
      self.steps_to_live_l2reg = 0
    if self.l2reg_theta is None:
      self.l2reg_theta = 0
    if self.l2reg_lambda is None:
      self.l2reg_lambda = 0
    if self.ht_type is None:
      self.ht_type = ""
    if self.emb_index is None:
      self.emb_index = 0
    if self.slot_index is None:
      self.slot_index = 0
