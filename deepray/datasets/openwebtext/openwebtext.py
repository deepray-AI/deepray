# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Helpers for preparing pre-training data and supplying them to the model."""

import sys
import multiprocessing

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline

FLAGS([
  sys.argv[0],
  "--num_train_examples=60000",
])


class Openwebtext(DataPipeline):
  def __init__(self, max_seq_length, **kwargs):
    super().__init__(**kwargs)
    self._max_seq_length = max_seq_length

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, *args, **kwargs):
    """The actual input function."""
    input_files = tf.io.gfile.glob(input_file_pattern)

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
    }

    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files))

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(multiprocessing.cpu_count(), len(input_files))

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
      tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, sloppy=is_training, cycle_length=cycle_length)
    )
    d = d.shuffle(buffer_size=100)

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don"t* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
      tf.data.experimental.map_and_batch(
        lambda record: self.parser(record, name_to_features),
        batch_size=batch_size,
        num_parallel_batches=multiprocessing.cpu_count(),
        drop_remainder=True,
      )
    )
    return d

  def parser(self, record, name_to_features):
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
