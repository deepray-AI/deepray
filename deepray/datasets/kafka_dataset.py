from tensorflow.python.data.ops import readers
import tensorflow as tf
from deepray.datasets.datapipeline import DataPipeLine
from absl import flags

FLAGS = flags.FLAGS


class KafkaDataset(DataPipeLine):

  def parse(self, raw_message, raw_key):
    context_features, sequence_features = {}, {}
    for key, dim in self.feature_map["FLOAT"].items():
      context_features[key] = tf.io.FixedLenFeature([], tf.float32)
    for key, dim in self.feature_map["INT"].items():
      context_features[key] = tf.io.FixedLenFeature([], tf.int64)
    for key, dim in self.feature_map["VARINT"].items():
      sequence_features[key] = tf.io.VarLenFeature(tf.int64)

    tensor, sparse_tensor = tf.io.parse_single_sequence_example(
        serialized=raw_message, context_features=context_features, sequence_features=sequence_features
    )
    reshaped_tensor = {}
    for fea in context_features:
      reshaped_tensor[fea] = tensor[fea]
      # reshaped_tensor[fea] = tf.reshape(tensor[fea], [1])
    label = reshaped_tensor.pop(FLAGS.label)
    for fea in sequence_features:
      reshaped_tensor[fea] = sparse_tensor[fea]
      # reshaped_tensor[fea] = tf.sparse.reshape(sparse_tensor[fea], [-1])
    return reshaped_tensor, label

  def build_dataset(self):
    dataset = (
        readers.KafkaGroupIODataset(
            topics=self.conf["Kafka"]["topics"],
            group_id=self.conf["Kafka"]["group_id"],
            servers=self.conf["Kafka"]["servers"],
            stream_timeout=3000,
            configuration=self.conf["Kafka"]["configuration"],
        ).map(map_func=self.parse, num_parallel_calls=FLAGS.parallel_parse).batch(FLAGS.batch_size)
    )
    return dataset
