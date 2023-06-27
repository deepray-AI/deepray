import tensorflow as tf
from deepray.datasets.datapipeline import DataPipeLine
from absl import flags

FLAGS = flags.FLAGS


class CSVPipeLine(DataPipeLine):
    def build_dataset(self, csv_path):
        dataset = tf.data.experimental.make_csv_dataset(
            csv_path,
            record_defaults=list(self.feature_map["dtype"]),
            column_names=list(self.feature_map["name"]),
            batch_size=FLAGS.batch_size,
            label_name=FLAGS.label,
            field_delim=",",
            header=True,
        )
        return dataset
