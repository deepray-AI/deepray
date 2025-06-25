import tensorflow as tf
from absl import flags
from tensorflow.python.data.ops import dataset_ops

from deepray.custom_ops.parquet_dataset import parquet_dataset_ops
from deepray.datasets.datapipeline import DataPipeline


class Ali_display_ad_click(DataPipeline):
  def parse(self, record):
    label_map = {}
    for label in flags.FLAGS.label:
      # label_map[label] = record.pop(label)
      label_map[label] = tf.reshape(record.pop(label), [-1, 1])
    return record, label_map

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, *args, **kwargs):
    """Makes dataset (of filenames) from filename glob patterns."""
    # Extract lines from input files using the Dataset API.

    file_list = self.read_list_from_file(input_file_pattern)

    dataset = parquet_dataset_ops.ParquetDataset(
      file_list,
      batch_size=batch_size,
      fields=[
        parquet_dataset_ops.DataFrame.Field(k, dtype, ragged_rank=1 if length != 1 else 0)
        for k, dtype, length in self.feature_map[["name", "dtype", "length"]].values
      ],
      num_parallel_reads=flags.FLAGS.parallel_reads_per_file
      if flags.FLAGS.parallel_reads_per_file
      else dataset_ops.AUTOTUNE,
    )
    dataset = dataset.map(
      map_func=self.parse,
      num_parallel_calls=flags.FLAGS.parallel_parse if flags.FLAGS.parallel_parse else dataset_ops.AUTOTUNE,
    )
    if flags.FLAGS.shuffle_buffer:
      dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=flags.FLAGS.shuffle_buffer, count=flags.FLAGS.epochs)
      )
    else:
      dataset = dataset.repeat(flags.FLAGS.epochs)
    return dataset
