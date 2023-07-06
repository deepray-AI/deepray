import tensorflow as tf
from deepray.datasets.datapipeline import DataPipeLine
from tensorflow.python.data.ops import dataset_ops

from absl import flags

FLAGS = flags.FLAGS


class TFRecordDataset(DataPipeLine):

  def __init__(self, compression_type=None, **kwargs):
    super().__init__(**kwargs)
    self.compression_type = compression_type

    self.context_features, self.sequence_features = {}, {}
    # for key, dtype in self.feature_map.loc[self.feature_map["length"] == 1][["name", "dtype"]].values:
    #   context_features[key] = tf.io.FixedLenFeature([FLAGS.prebatch], dtype)
    # for key, dtype in self.feature_map.loc[self.feature_map["length"] > 1][["name", "dtype"]].values:
    #   sequence_features[key] = tf.io.VarLenFeature(dtype)
    for key, dtype, length in self.feature_map[["name", "dtype", "length"]].values:
      self.context_features[key] = tf.io.FixedLenFeature([length], dtype)

  def parser(self, record):
    tensor, sparse_tensor, ragged_tensor = tf.io.parse_sequence_example(
        serialized=record, context_features=self.context_features, sequence_features=self.sequence_features
    )

    tensor.update(sparse_tensor)
    label_map = {}
    for label in FLAGS.label:
      label_map[label] = tensor.pop(label)
    return tensor, label_map

  def build_dataset(
      self, input_file_pattern, batch_size, is_training=True, prebatch_size=0, epochs=1, shuffle=True, *args, **kwargs
  ):
    file_list = self.read_list_from_file(input_file_pattern)
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    dataset = (
        tf.data.TFRecordDataset(
            file_list,
            compression_type=self.compression_type,
            num_parallel_reads=FLAGS.parallel_reads_per_file if FLAGS.parallel_reads_per_file else dataset_ops.AUTOTUNE,
            # automatically interleaves reads from multiple files
        ).batch(
            batch_size,
            # here will waste a little data, sparse_feature need to be reshaped with precise batch_size,
            # so drop remainder
            drop_remainder=True,
        ).map(
            map_func=self.parser,
            num_parallel_calls=FLAGS.parallel_parse if FLAGS.parallel_parse else dataset_ops.AUTOTUNE,
        )
    )
    # if flags.epochs > 1:
    #     dataset = dataset.cache()
    if FLAGS.shuffle_buffer:
      dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer)
    if not disable_cache:
      dataset = dataset.cache()
    return dataset
