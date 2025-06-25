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

import os
import shutil

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline

AUTOTUNE = tf.data.AUTOTUNE


class IMDB(DataPipeline):
  def __init__(self, url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", **kwargs):
    super().__init__(**kwargs)
    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir="")

    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

    train_dir = os.path.join(dataset_dir, "train")

    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, "unsup")
    shutil.rmtree(remove_dir)

  def parser(self, record):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=True, *args, **kwargs):
    if is_training:
      raw_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/train", batch_size=batch_size, seed=FLAGS.random_seed
      )

    else:
      raw_ds = tf.keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)

    raw_ds = raw_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return raw_ds
