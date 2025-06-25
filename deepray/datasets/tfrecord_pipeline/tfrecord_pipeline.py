import multiprocessing

import tensorflow as tf

from deepray.datasets.datapipeline import DataPipeline
from deepray.utils.horovod_utils import get_rank, get_world_size


class TFRecordPipeline(DataPipeline):
  """
  Build a pipeline fetching, shuffling, and preprocessing the tfrecord files.
  """

  def __init__(self, compression_type=None, **kwargs):
    super().__init__(**kwargs)
    self.compression_type = compression_type

  @property
  def features(self):
    context_features, sequence_features = {}, {}
    # for key, dtype in self.feature_map.loc[self.feature_map["length"] == 1][["name", "dtype"]].values:
    #   context_features[key] = tf.io.FixedLenFeature([FLAGS.prebatch], dtype)
    # for key, dtype in self.feature_map.loc[self.feature_map["length"] > 1][["name", "dtype"]].values:
    #   sequence_features[key] = tf.io.VarLenFeature(dtype)
    for key, dtype, length in self.feature_map[["name", "dtype", "length"]].values:
      context_features[key] = tf.io.FixedLenFeature([length], dtype)
    return context_features, sequence_features

  def parser(self, record):
    self.context_features, self.sequence_features = self.features
    tensor, sparse_tensor, ragged_tensor = tf.io.parse_sequence_example(
      serialized=record, context_features=self.context_features, sequence_features=self.sequence_features
    )

    tensor.update(sparse_tensor)
    label_map = {}
    for label in FLAGS.label:
      label_map[label] = tensor.pop(label)
    return tensor, label_map

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=True, *args, **kwargs):
    input_files = tf.io.gfile.glob(input_file_pattern)

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_files, str) or len(input_files) < get_world_size():
      dataset = tf.data.TFRecordDataset(
        input_files, compression_type=self.compression_type, num_parallel_reads=tf.data.AUTOTUNE
      )
      if self.use_horovod:
        # For multi-host training, we want each hosts to always process the same
        # subset of files.  Each host only sees a subset of the entire dataset,
        # allowing us to cache larger datasets in memory.
        dataset = dataset.shard(num_shards=get_world_size(), index=get_rank())

    else:
      dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      if self.use_horovod:
        # For multi-host training, we want each hosts to always process the same
        # subset of files.  Each host only sees a subset of the entire dataset,
        # allowing us to cache larger datasets in memory.
        dataset = dataset.shard(num_shards=get_world_size(), index=get_rank())

      def mfunc(x):
        rst = tf.data.TFRecordDataset(x, compression_type=self.compression_type)
        return rst

      # In parallel, create tf record dataset for each train files.
      # cycle_length = 8 means that up to 8 files will be read and deserialized in
      # parallel. You may want to increase this number if you have a large number of
      # CPU cores.
      cycle_length = min(multiprocessing.cpu_count(), len(input_files))
      dataset = dataset.interleave(
        mfunc, cycle_length=cycle_length, block_length=4, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
      )

    dataset = dataset.batch(batch_size).map(self.parser, multiprocessing.cpu_count())
    # dataset = dataset.ignore_errors()
    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.with_options(self._dataset_options(input_files))
    # Using `ignore_errors()` will drop the element that causes an error.
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset
