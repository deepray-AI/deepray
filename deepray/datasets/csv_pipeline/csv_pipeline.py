import tensorflow as tf
from deepray.datasets.datapipeline import DataPipeline
from absl import flags


class CSVPipeline(DataPipeline):
  def build_dataset(self, batch_size, input_file_pattern, is_training=True, epochs=1, shuffle=False, *args, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      input_file_pattern,
      record_defaults=list(self.feature_map["dtype"]),
      column_names=list(self.feature_map["name"]),
      batch_size=batch_size,
      label_name=flags.FLAGS.label,
      field_delim=",",
      header=True,
    )
    return dataset
