# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.tfrecord_pipeline import TFRecordPipeline

LABEL = ["label"]
NEGATIVE_HISTORY = ["item_feat_0_neg", "item_feat_1_neg"]
POSITIVE_HISTORY = ["item_feat_0_pos", "item_feat_1_pos"]
TARGET_ITEM_FEATURES = ["item_feat_0_trgt", "item_feat_1_trgt"]
USER_FEATURES = ["user_feat_0"]


class AmazonBooks2014(TFRecordPipeline):
  def __init__(self, max_seq_length, **kwargs):
    super().__init__(**kwargs)
    self._max_seq_length = max_seq_length
    flags.FLAGS([
      sys.argv[0],
      "--num_train_examples=11932672",
    ])

  def parser(self, record):
    tf_feature_spec = {
      name: tf.io.FixedLenFeature([length] if length > 1 else [], tf.int64)
      for name, dtype, length in self.feature_map[["name", "dtype", "length"]].values
    }

    sample = tf.io.parse_example(serialized=record, features=tf_feature_spec)

    user_features = {f_name: tf.reshape(sample[f_name], [-1]) for f_name in USER_FEATURES}

    target_item_features = {f_name: tf.reshape(sample[f_name], [-1]) for f_name in TARGET_ITEM_FEATURES}

    padded_positive = {
      f_name: tf.reshape(
        sample[f_name], [-1, self.feature_map.loc[self.feature_map["name"] == f_name, "length"].values[0]]
      )
      for f_name in POSITIVE_HISTORY
    }

    padded_negative = {
      f_name: tf.reshape(
        sample[f_name], [-1, self.feature_map.loc[self.feature_map["name"] == f_name, "length"].values[0]]
      )
      for f_name in NEGATIVE_HISTORY
    }

    long_sequence_features = {f_name: val[:, : self._max_seq_length] for f_name, val in padded_positive.items()}

    short_sequence_features = {f_name: val[:, self._max_seq_length :] for f_name, val in padded_positive.items()}

    short_neg_sequence_features = {f_name: val[:, self._max_seq_length :] for f_name, val in padded_negative.items()}

    first_positive_feature_name = POSITIVE_HISTORY[0]
    first_positive_feature = padded_positive[first_positive_feature_name]

    history_mask = tf.cast(tf.greater(first_positive_feature, 0), tf.float32)

    long_sequence_mask = history_mask[:, : self._max_seq_length]
    short_sequence_mask = history_mask[:, self._max_seq_length :]

    label_name = LABEL[0]
    target = tf.reshape(sample[label_name], [-1])

    return {
      "user_features": user_features,
      "target_item_features": target_item_features,
      "long_sequence_features": long_sequence_features,
      "short_sequence_features": short_sequence_features,
      "short_neg_sequence_features": short_neg_sequence_features,
      "long_sequence_mask": long_sequence_mask,
      "short_sequence_mask": short_sequence_mask,
      "other_features": None,
    }, target
