from typing import List

import tensorflow as tf
import tensorflow_io as tfio
from absl import flags

from deepray.datasets.datapipeline import DataPipeLine

FLAGS = flags.FLAGS


class parquet_pipeline(DataPipeLine):
  def __init__(self, *args, **kwargs):
    super(parquet_pipeline, self).__init__(*args, **kwargs)
    self._column_names = ["fe_" + str(x) for x in list(self.feature_map[~(self.feature_map['ftype'] == "Label")]["code"])]
    self._column_names.append("detailed_deal_type")

    self.columns = {
      "detailed_deal_type": tf.TensorSpec(tf.TensorShape([]), tf.string),
      "fe_3544": tf.TensorSpec(tf.TensorShape([]), tf.float32),
      "fe_1098": tf.TensorSpec(tf.TensorShape([]), tf.int64),
      "fe_777": tf.TensorSpec(tf.TensorShape([]), tf.float32),
      'fe_2604.list.element': tf.TensorSpec(tf.TensorShape([30]), tf.int64),

      # "fe_3544": tf.TensorSpec(tf.TensorShape([]), tf.int64),

    }

    self._output_types = self.get_tf_types(list(self.feature_map["dtype"]))
    self._output_shapes = tuple(self.feature_map["length"])

    def remove_prefix(s: str) -> str:
      prefix = "f_"
      if s.startswith(prefix):
        return s[len(prefix):]
      else:
        return s

    self.__key_names = [remove_prefix(x) for x in self._column_names]

  def get_tf_types(self, str_type: List[str]):
    tf_type_map = {
      "int32": tf.dtypes.int32,
      "int64": tf.dtypes.int64,
      "float32": tf.dtypes.float32,
      "float64": tf.dtypes.float64,
      "double": tf.dtypes.float64,
      "long": tf.dtypes.int64,
      "string": tf.dtypes.string
    }
    return [tf_type_map[x] for x in str_type]

  def build_dataset(self, input_file_pattern,
                    batch_size,
                    is_training=True,
                    prebatch_size=0,
                    *args, **kwargs):
    filename = tf.io.gfile.glob(input_file_pattern)

    gpu_transform = tf.data.experimental.prefetch_to_device('GPU:0', buffer_size=10)

    def map_fn(file_location):
      return tfio.IODataset.from_parquet(file_location, columns=self.columns)

    dataset = tf.data.Dataset.list_files(filename).map(map_fn)
    # dataset = tfio.IODataset.from_parquet(filename, self._column_names) \
    #   .map(
    #     self.parser,
    #     multiprocessing.cpu_count()
    #     # num_parallel_calls=tf.data.AUTOTUNE
    #   )
    #   .prefetch(100)
    #   .apply(gpu_transform)
    # )
    return dataset

  @tf.function
  def parser(self, *features):
    sample = dict(zip(self._column_names_v2, features))
    # 直接返回所有sample，后续处理在gpu进行
    # sample = {}

    # predict hdd
    # for i in range(len(features) - 1):
    #   sample[self._column_names[i][3:]] = features[i]  # parquet的列名不能是数字，所以给所有表头也就是特征编号加了个f_，现在要剔除
    # sample[self._column_names[len(features) - 1]] = features[len(features) - 1]

    with tf.name_scope("train_action"):
      # train_action = sample[self.train_conf.multi_task_conf.train_action_key]
      train_action = sample.pop(self.train_conf.multi_task_conf.train_action_key)
      labels = {}
      weights = {}
      for task in self.task_names:
        task_w = self.task_train_weight[task].lookup(train_action)
        labels[task] = tf.cast(tf.math.greater(task_w, 0), tf.int32)
        weights[task] = tf.math.abs(task_w)
    return sample, labels, weights
