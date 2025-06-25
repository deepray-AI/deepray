import random

import pandas as pd
import tensorflow as tf
from absl import flags
from six import string_types
from tensorflow import dtypes
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

from deepray.custom_ops.parquet_dataset import parquet_dataset_ops
from deepray.custom_ops.parquet_dataset.python.parquet_pybind import parquet_filenames_and_fields
from deepray.datasets.datapipeline import DataPipeline
from deepray.utils import logging_util
from deepray.utils.horovod_utils import get_rank, get_world_size

logger = logging_util.get_logger()


def parquet_filenames(filenames, lower=False):
  """Check and fetch parquet filenames and fields.

  Args:
    filenames: List of Path of parquet file list.
    lower: Convert field name to lower case if not found.

  Returns:
    Validated file names and fields.
  """
  if isinstance(filenames, string_types):
    filenames = [filenames]
  elif isinstance(filenames, (tuple, list)):
    for f in filenames:
      if not isinstance(f, string_types):
        raise ValueError(f"{f} in `filenames` must be a string")
  elif isinstance(filenames, dataset_ops.Dataset):
    if filenames.output_types != dtypes.string:
      raise TypeError("`filenames` must be a `tf.data.Dataset` of `tf.string` elements.")
    if not filenames.output_shapes.is_compatible_with(tensor_shape.TensorShape([])):
      raise ValueError("`filenames` must be a `tf.data.Dataset` of scalar `tf.string` elements.")
  elif isinstance(filenames, ops.Tensor):
    if filenames.dtype != dtypes.string:
      raise TypeError("`filenames` must be a `tf.Tensor` of `tf.string`.")
  else:
    raise ValueError(
      f"`filenames` {filenames} must be a `tf.data.Dataset` of scalar "
      "`tf.string` elements or can be converted to a `tf.Tensor` of "
      "`tf.string`."
    )

  if not isinstance(filenames, dataset_ops.Dataset):
    filenames = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filenames = array_ops.reshape(filenames, [-1], name="filenames")
    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)
  return filenames


class ParquetDataset(dataset_ops.DatasetV2):  # pylint: disable=abstract-method
  """A Parquet Dataset that reads batches from parquet files."""

  VERSION = 2002

  def __init__(
    self, filenames, column_names=None, batch_size=1, num_parallel_reads=None, num_sequential_reads=2, parser=None
  ):
    """Create a `ParquetDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      batch_size: (Optional.) Maxium number of samples in an output batch.
      column_names: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
    """
    self._batch_size = batch_size
    self._filter = filter
    self._parser = parser

    filenames, fields = parquet_filenames_and_fields(filenames, column_names)
    filenames = filenames.batch(32)

    def _create_dataset(f):
      dataset = parquet_dataset_ops.ParquetDataset(
        filenames=f,
        fields=fields,
        batch_size=self._batch_size,
      )
      if self._parser:
        dataset = dataset.map(self._parser, num_parallel_calls=tf.data.AUTOTUNE)
      return dataset

    self._impl = self._build_dataset(
      _create_dataset, filenames, num_parallel_reads=num_parallel_reads, num_sequential_reads=num_sequential_reads
    )
    super().__init__(self._impl._variant_tensor)  # pylint: disable=protected-access

  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def element_spec(self):
    return self._impl.element_spec  # pylint: disable=protected-access

  def _build_dataset(self, dataset_creator, filenames, num_parallel_reads=None, num_sequential_reads=1):
    """Internal method to create a `ParquetDataset`."""
    if num_parallel_reads is None:
      return filenames.flat_map(dataset_creator)
    if num_parallel_reads == dataset_ops.AUTOTUNE:
      return filenames.interleave(dataset_creator, num_parallel_calls=2, deterministic=False)
    return readers.ParallelInterleaveDataset(
      filenames,
      dataset_creator,
      cycle_length=num_parallel_reads,
      block_length=num_sequential_reads,
      sloppy=True,
      buffer_output_elements=None,
      prefetch_input_elements=1,
    )


class ParquetPipeline(DataPipeline):
  def __init__(self, column_names=[], **kwargs):
    super().__init__(**kwargs)
    self.column_names = column_names

    # duplicate value check
    visited = set()
    dup_values = [name for name in self.column_names if name in visited or (visited.add(name) or False)]
    assert len(dup_values) == 0, "The column_names input parameter has duplicate values: " + str(dup_values)

    self.info_df = pd.DataFrame()

  def parse(self, record):
    label_map = {}
    for label in flags.FLAGS.label:
      # label_map[label] = record.pop(label)
      label_map[label] = tf.reshape(record.pop(label), [-1, 1])
    return record, label_map

  def build_dataset(
    self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ) -> tf.data.Dataset:
    if isinstance(input_file_pattern, str):
      data_file_list = self.read_list_from_file(input_file_pattern)
    else:
      data_file_list = input_file_pattern
    if not data_file_list:
      raise ValueError("The input file list is empty!")

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    random_state = flags.FLAGS.random_seed if flags.FLAGS.random_seed else 1024
    if shuffle and isinstance(data_file_list, list):
      random.Random(random_state).shuffle(data_file_list)
      logger.info(f"Shuffling {len(data_file_list)} parquet files.")
    if isinstance(data_file_list, str) or len(data_file_list) < get_world_size():
      dataset = parquet_dataset_ops.ParquetDataset(
        filenames=data_file_list,
        fields=self.column_names if self.column_names else None,
        batch_size=batch_size,
      )
      if self.use_horovod:
        # For multi-host training, we want each hosts to always process the same
        # subset of files.  Each host only sees a subset of the entire dataset,
        # allowing us to cache larger datasets in memory.
        dataset = dataset.shard(num_shards=get_world_size(), index=get_rank())
        logger.info("Using samples distributing strategy ❤")
      if not hasattr(self.parser, "__isabstractmethod__"):
        dataset = dataset.map(self.parser, tf.data.AUTOTUNE)
    else:
      if self.use_horovod:
        # For multi-host training, we want each hosts process different
        # subset of files.  Each host only sees a subset of the entire dataset,
        # allowing us to cache larger datasets in memory.
        data_file_list = [data_file_list[i] for i in range(len(data_file_list)) if i % get_world_size() == get_rank()]
        logger.info("Using files distributing strategy ❤")
      dataset = ParquetDataset(
        filenames=data_file_list,
        column_names=self.column_names if self.column_names else None,
        batch_size=batch_size,
        num_parallel_reads=dataset_ops.AUTOTUNE,
        parser=None if hasattr(self.parser, "__isabstractmethod__") else self.parser,
      )

    # if not hasattr(self.parser, "__isabstractmethod__"):
    #   dataset = dataset.map(self.parser, multiprocessing.cpu_count())
    # dataset = dataset.ignore_errors()
    # Prefetch overlaps in-feed with training
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.with_options(self._dataset_options(data_file_list))
    # Using `ignore_errors()` will drop the element that causes an error.
    # dataset = dataset.apply(tf.data.experimental.ignore_errors())

    if shuffle:
      shuffle_buffer = kwargs.get("shuffle_buffer", 10)
      logger.debug(f"kwargs = {kwargs}")
      logger.info(f"The shuffle_buffer is {shuffle_buffer}")
      dataset = (
        dataset.unbatch()
        .shuffle(buffer_size=shuffle_buffer, seed=flags.FLAGS.random_seed, reshuffle_each_iteration=False)
        .batch(batch_size)
      )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
