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
from tensorflow.python.data.ops import dataset_ops

import tensorflow as tf
from absl import flags
import horovod.tensorflow as hvd

from deepray.datasets.datapipeline import DataPipeLine

FLAGS = flags.FLAGS
FLAGS([
    sys.argv[0],
    "--num_train_examples=24324736",
])


class Wikicorpus_en(DataPipeLine):

  def __init__(self, max_seq_length, **kwargs):
    super().__init__(**kwargs)
    self._max_seq_length = max_seq_length

    self.name_to_features = {
        "input_ids": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([self._max_seq_length], tf.int64),
    }

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, prebatch_size=0, *args, **kwargs):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    input_files = tf.io.gfile.glob(input_file_pattern)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    if self.use_horovod:
      dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
    dataset = dataset.repeat(FLAGS.epochs)
    dataset = dataset.shuffle(buffer_size=len(input_files), seed=FLAGS.random_seed, reshuffle_each_iteration=False)

    cycle_length = min(multiprocessing.cpu_count(), len(input_files))
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length, deterministic=True).shuffle(
        buffer_size=100, seed=FLAGS.random_seed, reshuffle_each_iteration=False
    ).map(
        map_func=self.parser,
        num_parallel_calls=FLAGS.parallel_parse if FLAGS.parallel_parse else dataset_ops.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)

    return dataset

  def parser(self, record):
    """Decodes a record to a TensorFlow example."""

    example = tf.io.parse_single_example(record, self.name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example
