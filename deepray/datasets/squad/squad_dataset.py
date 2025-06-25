# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================
"""BERT model input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepray.datasets.datapipeline import DataPipeline
from deepray.utils.horovod_utils import get_rank, get_world_size


class Squad(DataPipeline):
  def __init__(self, max_seq_length, input_pipeline_context=None, **kwargs):
    super().__init__(**kwargs)
    self.max_seq_length = max_seq_length
    self.input_pipeline_context = input_pipeline_context

  def decode_record(self, record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def single_file_dataset(self, input_file, name_to_features):
    """Creates a single-file dataset to be passed for BERT custom training."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if self.use_horovod:
      d = d.shard(num_shards=get_world_size(), index=get_rank())

    d = d.map(lambda record: self.decode_record(record, name_to_features))

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_file, str) or len(input_file) == 1:
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
      d = d.with_options(options)
    return d

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=False, *args, **kwargs):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
    }
    if is_training:
      name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
      name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
    else:
      name_to_features["unique_ids"] = tf.io.FixedLenFeature([], tf.int64)

    dataset = self.single_file_dataset(input_file_pattern, name_to_features)

    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if self.input_pipeline_context and self.input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(
        self.input_pipeline_context.num_input_pipelines, self.input_pipeline_context.input_pipeline_id
      )

    def parser(record):
      """Dispatches record to features and labels."""
      x, y = {}, {}
      for name, tensor in record.items():
        if name in ("start_positions", "end_positions"):
          y[name] = tensor
        elif name == "input_ids":
          x["input_word_ids"] = tensor
        elif name == "segment_ids":
          x["input_type_ids"] = tensor
        else:
          x[name] = tensor
      return x, y

    dataset = dataset.map(parser)

    if is_training:
      dataset = dataset.shuffle(100)
      # dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1024)
    return dataset
