# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data pipeline for the Ranking model.

This module defines various input datasets for the Ranking model.

https://github.com/tensorflow/models/blob/master/official/recommendation/ranking/data/data_pipeline.py
"""

from typing import List
import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeLine

flags.DEFINE_bool("sharding", True, "Whether sharding is used in the input pipeline.")
flags.DEFINE_integer("num_shards_per_host", 4, "Number of shards per host.")
FLAGS = flags.FLAGS

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
  'C1': 2500,
  'C2': 2000,
  'C3': 5000000,
  'C4': 1500000,
  'C5': 1000,
  'C6': 100,
  'C7': 20000,
  'C8': 4000,
  'C9': 20,
  'C10': 100000,
  'C11': 10000,
  'C12': 5000000,
  'C13': 40000,
  'C14': 100,
  'C15': 100,
  'C16': 3000000,
  'C17': 50,
  'C18': 10000,
  'C19': 4000,
  'C20': 20,
  'C21': 4000000,
  'C22': 100,
  'C23': 100,
  'C24': 250000,
  'C25': 400,
  'C26': 100000
}


class CriteoTsvReader(DataPipeLine):
  """Input reader callable for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.
  """

  def __init__(self, num_dense_features: int, vocab_sizes: List[int], use_synthetic_data: bool = False):
    super().__init__()
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._use_synthetic_data = use_synthetic_data

  @tf.function
  def parser(self, record: tf.Tensor):
    """Parser function for pre-processed Criteo TSV records."""
    # label_defaults = [[0.0]]
    # dense_defaults = [
    #   [0.0] for _ in range(self._num_dense_features)
    # ]
    # num_sparse_features = len(self._vocab_sizes)
    # categorical_defaults = [
    #   [0] for _ in range(num_sparse_features)
    # ]
    # record_defaults = label_defaults + dense_defaults + categorical_defaults
    # fields = tf.io.decode_csv(
    #   record, record_defaults, field_delim='\t', na_value='-1')

    dense_defaults = [[0.0] for i in range(1, 14)]
    cate_defaults = [[' '] for i in range(1, 27)]
    num_sparse_features = len(cate_defaults)
    label_defaults = [[0]]
    column_headers = TRAIN_DATA_COLUMNS
    record_defaults = label_defaults + dense_defaults + cate_defaults
    fields = tf.io.decode_csv(record, record_defaults=record_defaults, na_value='-1')

    num_labels = 1
    label = tf.reshape(fields[0], [FLAGS.batch_size, 1])

    features = {}
    num_dense = len(dense_defaults)

    dense_features = []
    offset = num_labels
    for idx in range(num_dense):
      dense_features.append(fields[idx + offset])
    features['dense_features'] = tf.stack(dense_features, axis=1)

    offset += num_dense
    features['sparse_features'] = {}

    for idx in range(num_sparse_features):
      features['sparse_features'][str(idx)] = fields[idx + offset]

    return features, label

  def _generate_synthetic_data(self, ctx: tf.distribute.InputContext,
                               batch_size: int,
                               is_training=True) -> tf.data.Dataset:
    """Creates synthetic data based on the parameter batch size.

    Args:
      ctx: Input Context
      batch_size: per replica batch size.

    Returns:
      The synthetic dataset.
    """
    num_dense = self._num_dense_features
    num_replicas = ctx.num_replicas_in_sync if ctx else 1

    if is_training:
      dataset_size = 1000 * batch_size * num_replicas
    else:
      dataset_size = 1000 * batch_size * num_replicas
    dense_tensor = tf.random.uniform(
      shape=(dataset_size, num_dense), maxval=1.0, dtype=tf.float32)

    sparse_tensors = []
    for size in self._vocab_sizes:
      sparse_tensors.append(
        tf.random.uniform(
          shape=(dataset_size,), maxval=int(size), dtype=tf.int32))

    sparse_tensor_elements = {
      str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
    }

    # the mean is in [0, 1] interval.
    dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)

    sparse_tensors = tf.stack(sparse_tensors, axis=-1)
    sparse_tensors_mean = tf.math.reduce_sum(sparse_tensors, axis=1)
    # the mean is in [0, 1] interval.
    sparse_tensors_mean = tf.cast(sparse_tensors_mean, dtype=tf.float32)
    sparse_tensors_mean /= sum(self._vocab_sizes)
    # the label is in [0, 1] interval.
    label_tensor = (dense_tensor_mean + sparse_tensors_mean) / 2.0
    # Using the threshold 0.5 to convert to 0/1 labels.
    label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

    input_elem = {'dense_features': dense_tensor,
                  'sparse_features': sparse_tensor_elements}, label_tensor

    dataset = tf.data.Dataset.from_tensor_slices(input_elem)
    dataset = dataset.cache()
    if is_training:
      dataset = dataset.repeat()

    return dataset.batch(batch_size, drop_remainder=True)

  def build_dataset(self, input_file_pattern: str,
                    batch_size,
                    is_training=True,
                    context: tf.distribute.InputContext = None,
                    use_horovod=False,
                    *args, **kwargs):
    if self._use_synthetic_data:
      return self._generate_synthetic_data(context, batch_size, is_training)

    filenames = tf.data.Dataset.list_files(input_file_pattern, shuffle=False)

    # Shard the full dataset according to host number.
    # Each host will get 1 / num_of_hosts portion of the data.
    if FLAGS.sharding and context and context.num_input_pipelines > 1:
      filenames = filenames.shard(context.num_input_pipelines,
                                  context.input_pipeline_id)

    num_shards_per_host = 1
    if FLAGS.sharding:
      num_shards_per_host = FLAGS.num_shards_per_host

    def make_dataset(shard_index):
      filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
      dataset = tf.data.TextLineDataset(filenames_for_shard)
      if is_training:
        dataset = dataset.repeat()
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.map(self.parser,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    indices = tf.data.Dataset.range(num_shards_per_host)
    dataset = indices.interleave(
      map_func=make_dataset,
      cycle_length=FLAGS.interleave_cycle,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
