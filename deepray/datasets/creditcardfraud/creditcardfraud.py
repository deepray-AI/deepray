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
"""Credit Card Fraud dataset."""

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deepray.datasets.datapipeline import DataPipeline

flags.FLAGS([
  sys.argv[0],
  "--num_train_examples=182280",
])


class CreditCardFraud(DataPipeline):
  def __init__(self, url="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"):
    super().__init__()
    csv_file = tf.keras.utils.get_file("creditcard.csv", url)
    raw_df = pd.read_csv(csv_file)

    raw_df[["Time", "V1", "V2", "V3", "V4", "V5", "V26", "V27", "V28", "Amount", "Class"]].describe()

    neg, pos = np.bincount(raw_df["Class"])
    total = neg + pos
    print("Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(total, pos, 100 * pos / total))

    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    cleaned_df.pop("Time")

    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df["Log Amount"] = np.log(cleaned_df.pop("Amount") + eps)

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    self.train_labels = np.array(train_df.pop("Class"))
    self.bool_train_labels = self.train_labels != 0
    self.val_labels = np.array(val_df.pop("Class"))
    self.test_labels = np.array(test_df.pop("Class"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    self.train_features = np.clip(train_features, -5, 5)
    self.val_features = np.clip(val_features, -5, 5)
    self.test_features = np.clip(test_features, -5, 5)

    self.train_df = pd.DataFrame(self.train_features, columns=train_df.columns)
    self.val_df = pd.DataFrame(self.val_features, columns=train_df.columns)

  def __len__(self):
    pass

  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    if is_training:
      ds = tf.data.Dataset.from_tensor_slices((self.train_features, self.train_labels))

    else:
      ds = tf.data.Dataset.from_tensor_slices((self.val_features, self.val_labels))
    ds = ds.repeat(flags.FLAGS.epochs).shuffle(10000).batch(batch_size)
    return ds
