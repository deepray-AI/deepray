# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Asynchronous data producer for the NCF pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import os
import sys
import tempfile
import threading
import time
import timeit
import traceback
import typing

from absl import logging
import numpy as np
from six.moves import queue
import tensorflow as tf
from . import movielens
from . import popen_helper
from . import constants as rconst

from tensorflow.python.tpu.datasets import StreamingFilesDataset

import sys

from absl import flags

from deepray.datasets.datapipeline import DataPipeLine

FLAGS = flags.FLAGS


class Movielens(DataPipeLine):

  def __init__(
      self,
      stream_files: bool = False,
      use_synthetic_data: bool = False,
      shard_root=None,
      deterministic=False,
      **kwargs
  ):
    """Constructs a `Movielens DatasetManager` instance.

    Args:
      is_training: Boolean of whether the data provided is training or
        evaluation data. This determines whether to reuse the data (if
        is_training=False) and the exact structure to use when storing and
        yielding data.
      stream_files: Boolean indicating whether data should be serialized and
        written to file shards.
      batches_per_epoch: The number of batches in a single epoch.
      shard_root: The base directory to be used when stream_files=True.
      deterministic: Forgo non-deterministic speedups. (i.e. sloppy=True)
      num_train_epochs: Number of epochs to generate. If None, then each call to
        `get_dataset()` increments the number of epochs requested.
    """
    super().__init__(**kwargs)
    self._stream_files = stream_files
    self._use_synthetic_data = use_synthetic_data
    self._epochs_requested = FLAGS.epochs if FLAGS.epochs else 0
    self._shard_root = shard_root
    self._deterministic = deterministic

    self._result_queue = queue.Queue()
    self._result_reuse = []

  @property
  def current_data_root(self):
    subdir = (rconst.TRAIN_FOLDER_TEMPLATE.format(self._epochs_completed) if self._is_training else rconst.EVAL_FOLDER)
    return os.path.join(self._shard_root, subdir)

  @staticmethod
  def parser(serialized_data, batch_size=None, is_training=True):
    """Convert serialized TFRecords into tensors.

    Args:
      serialized_data: A tensor containing serialized records.
      batch_size: The data arrives pre-batched, so batch size is needed to
        deserialize the data.
      is_training: Boolean, whether data to deserialize to training data or
        evaluation data.
    """

    def _get_feature_map(batch_size, is_training=True):
      """Returns data format of the serialized tf record file."""

      if is_training:
        return {
            movielens.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
            movielens.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
            rconst.VALID_POINT_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
            "labels": tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64)
        }
      else:
        return {
            movielens.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
            movielens.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
            rconst.DUPLICATE_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64)
        }

    features = tf.io.parse_single_example(serialized_data, _get_feature_map(batch_size, is_training=is_training))
    users = tf.cast(features[movielens.USER_COLUMN], rconst.USER_DTYPE)
    items = tf.cast(features[movielens.ITEM_COLUMN], rconst.ITEM_DTYPE)

    if is_training:
      valid_point_mask = tf.cast(features[rconst.VALID_POINT_MASK], tf.bool)
      fake_dup_mask = tf.zeros_like(users)
      return {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
          rconst.VALID_POINT_MASK: valid_point_mask,
          rconst.TRAIN_LABEL_KEY: tf.reshape(tf.cast(features["labels"], tf.bool), (batch_size, 1)),
          rconst.DUPLICATE_MASK: fake_dup_mask
      }
    else:
      labels = tf.cast(tf.zeros_like(users), tf.bool)
      fake_valid_pt_mask = tf.cast(tf.zeros_like(users), tf.bool)
      return {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
          rconst.DUPLICATE_MASK: tf.cast(features[rconst.DUPLICATE_MASK], tf.bool),
          rconst.VALID_POINT_MASK: fake_valid_pt_mask,
          rconst.TRAIN_LABEL_KEY: labels
      }

  def data_generator(self, epochs, is_training):
    """Yields examples during local training."""
    assert not self._stream_files
    assert is_training or epochs == 1

    if is_training:
      for _ in range(self._batches_per_epoch * epochs):
        yield self._result_queue.get(timeout=300)

    else:
      if self._result_reuse:
        assert len(self._result_reuse) == self._batches_per_epoch

        for i in self._result_reuse:
          yield i
      else:
        # First epoch.
        for _ in range(self._batches_per_epoch * epochs):
          result = self._result_queue.get(timeout=300)
          self._result_reuse.append(result)
          yield result

  def increment_request_epoch(self):
    self._epochs_requested += 1

  def build_dataset(
      self, input_file_pattern, batch_size, is_training=True, prebatch_size=0, epochs=1, shuffle=True, *args, **kwargs
  ):
    """Construct the dataset to be used for training and eval.

    For local training, data is provided through Dataset.from_generator. For
    remote training (TPUs) the data is first serialized to files and then sent
    to the TPU through a StreamingFilesDataset.

    Args:
      batch_size: The per-replica batch size of the dataset.
      is_training: Boolean of whether the data provided is training or
        evaluation data. This determines whether to reuse the data (if
        is_training=False) and the exact structure to use when storing and
        yielding data.
      epochs: How many epochs worth of data to yield. (Generator
        mode only.)
    """
    self.increment_request_epoch()
    if self._stream_files:
      if epochs > 1:
        raise ValueError("epochs > 1 not supported for file "
                         "based dataset.")
      epoch_data_dir = self._result_queue.get(timeout=300)
      if not is_training:
        self._result_queue.put(epoch_data_dir)  # Eval data is reused.

      file_pattern = os.path.join(epoch_data_dir, rconst.SHARD_TEMPLATE.format("*"))
      dataset = StreamingFilesDataset(
          files=file_pattern,
          worker_job=popen_helper.worker_job(),
          num_parallel_reads=rconst.NUM_FILE_SHARDS,
          num_epochs=1,
          sloppy=not self._deterministic
      )
      map_fn = functools.partial(self.parser, batch_size=batch_size, is_training=is_training)
      dataset = dataset.map(map_fn, num_parallel_calls=16)

    else:
      types = {movielens.USER_COLUMN: rconst.USER_DTYPE, movielens.ITEM_COLUMN: rconst.ITEM_DTYPE}
      shapes = {
          movielens.USER_COLUMN: tf.TensorShape([batch_size, 1]),
          movielens.ITEM_COLUMN: tf.TensorShape([batch_size, 1])
      }

      if is_training:
        types[rconst.VALID_POINT_MASK] = bool
        shapes[rconst.VALID_POINT_MASK] = tf.TensorShape([batch_size, 1])

        types = (types, bool)
        shapes = (shapes, tf.TensorShape([batch_size, 1]))

      else:
        types[rconst.DUPLICATE_MASK] = bool
        shapes[rconst.DUPLICATE_MASK] = tf.TensorShape([batch_size, 1])

      data_generator = functools.partial(self.data_generator, epochs=epochs, is_training=is_training)
      dataset = tf.data.Dataset.from_generator(generator=data_generator, output_types=types, output_shapes=shapes)

    return dataset.prefetch(16)
