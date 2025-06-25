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

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline
from deepray.utils.horovod_utils import get_world_size, get_rank


class CriteoTsvReader(DataPipeline):
  """Input reader callable for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.
  """

  def __init__(self, file_pattern: str = None, use_synthetic_data: bool = False, **kwargs):
    super().__init__(**kwargs)
    self._file_pattern = file_pattern
    self._num_dense_features = self.feature_map["ftype"].value_counts()["Numerical"]
    self._vocab_sizes = self.feature_map[(self.feature_map["ftype"] == "Categorical")]["voc_size"].tolist()
    self._use_synthetic_data = use_synthetic_data

  def build_dataset(
    self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=True, *args, **kwargs
  ) -> tf.data.Dataset:
    if self._use_synthetic_data:
      return self._generate_synthetic_data(is_training, batch_size)

    filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    # Shard the full dataset according to host number.
    # Each host will get 1 / num_of_hosts portion of the data.
    if self.use_horovod:
      filenames = filenames.shard(get_world_size(), get_rank())

    def make_dataset():
      filenames_for_shard = filenames.shard(get_world_size(), get_rank())
      dataset = tf.data.TextLineDataset(filenames_for_shard)
      if is_training:
        dataset = dataset.repeat()
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.map(self.parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    indices = tf.data.Dataset.range(get_world_size())
    dataset = indices.interleave(
      map_func=make_dataset, cycle_length=flags.FLAGS.cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def parser(self, example: tf.Tensor):
    """Parser function for pre-processed Criteo TSV records."""
    label_defaults = [[0.0]]
    dense_defaults = [[0.0] for _ in range(self._num_dense_features)]
    num_sparse_features = len(self._vocab_sizes)
    categorical_defaults = [[0] for _ in range(num_sparse_features)]
    record_defaults = label_defaults + dense_defaults + categorical_defaults
    fields = tf.io.decode_csv(example, record_defaults, field_delim="\t", na_value="-1")

    num_labels = 1
    label = tf.reshape(fields[0], [flags.FLAGS.batch_size, 1])

    features = {}
    num_dense = len(dense_defaults)

    dense_features = []
    offset = num_labels
    for idx in range(num_dense):
      dense_features.append(fields[idx + offset])
    features["dense_features"] = tf.stack(dense_features, axis=1)

    offset += num_dense
    features["sparse_features"] = {}

    for idx in range(num_sparse_features):
      features["sparse_features"][str(idx)] = fields[idx + offset]

    return features, label

  def _generate_synthetic_data(self, is_training: bool, batch_size: int) -> tf.data.Dataset:
    """Creates synthetic data based on the parameter batch size.

    Args:
      ctx: Input Context
      batch_size: per replica batch size.

    Returns:
      The synthetic dataset.
    """
    num_dense = self._num_dense_features
    num_replicas = get_world_size()
    dataset_size = 1000 * batch_size * num_replicas
    dense_tensor = tf.random.uniform(shape=(dataset_size, num_dense), maxval=1.0, dtype=tf.float32)

    sparse_tensors = []
    sparse_tensor_elements = {}
    for name, voc_size, dtype in self.feature_map[(self.feature_map["ftype"] == "Categorical")][
      ["name", "voc_size", "dtype"]
    ].values:
      _tensor = tf.random.uniform(shape=(dataset_size,), maxval=int(voc_size), dtype=dtype)
      sparse_tensors.append(_tensor)
      sparse_tensor_elements[name] = _tensor

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

    sparse_tensor_elements.update({"dense_features": dense_tensor})

    input_elem = sparse_tensor_elements, label_tensor

    dataset = tf.data.Dataset.from_tensor_slices(input_elem)
    dataset = dataset.cache()
    if is_training:
      dataset = dataset.repeat()

    return dataset.batch(batch_size, drop_remainder=True)
