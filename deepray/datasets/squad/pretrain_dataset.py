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

  def build_dataset(
    self,
    input_file_pattern,
    batch_size,
    max_predictions_per_seq,
    is_training=True,
    epochs=1,
    shuffle=False,
    *args,
    **kwargs,
  ):
    """Creates input dataset from (tf)records files for pretraining."""
    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
      "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    dataset = tf.data.Dataset.list_files(input_file_pattern, shuffle=is_training)
    if self.use_horovod:
      dataset = dataset.shard(num_shards=get_world_size(), index=get_rank())

    if self.input_pipeline_context and self.input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(
        self.input_pipeline_context.num_input_pipelines, self.input_pipeline_context.input_pipeline_id
      )

    dataset = dataset.repeat()

    # We set shuffle buffer to exactly match total number of
    # training files to ensure that training data is well shuffled.
    input_files = []
    for input_pattern in input_file_pattern:
      input_files.extend(tf.io.gfile.glob(input_pattern))
    dataset = dataset.shuffle(len(input_files))

    # In parallel, create tf record dataset for each train files.
    # cycle_length = 8 means that up to 8 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
      tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    decode_fn = lambda record: self.decode_record(record, name_to_features)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def parser(record):
      """Filter out features to use for pretraining."""
      x = {
        "input_word_ids": record["input_ids"],
        "input_mask": record["input_mask"],
        "input_type_ids": record["segment_ids"],
        "masked_lm_positions": record["masked_lm_positions"],
        "masked_lm_ids": record["masked_lm_ids"],
        "masked_lm_weights": record["masked_lm_weights"],
        "next_sentence_labels": record["next_sentence_labels"],
      }

      y = record["masked_lm_weights"]

      return x, y

    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      dataset = dataset.shuffle(100)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1024)
    return dataset
