# -*- coding:utf-8 -*-
# Copyright 2019 The Jarvis Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.parquet_pipeline.ali_parquet_dataset import ParquetPipeline

dir_path = os.path.dirname(os.path.realpath(__file__))
FLAGS([
  sys.argv[0],
  "--num_train_examples=36210028",
])
if os.path.exists(os.path.join(dir_path, "feature_map.csv")):
  FLAGS([
    sys.argv[0],
    f"--feature_map={dir_path}/feature_map.csv",
  ])

DEFAULT_VALUE = {"int64": 0, "float32": 0.0, "bytes": ""}


class Avazu(ParquetPipeline):
  def parse(self, record):
    for name in self.feature_map[(self.feature_map["length"] == 1)]["name"].values:
      record[name] = tf.expand_dims(record[name], axis=1)

    if len(self.feature_map[(self.feature_map["ftype"] == "Label")].index) == 1:
      target = record.pop(self.feature_map[(self.feature_map["ftype"] == "Label")].iloc[0]["name"])
    else:
      target = {}
      for name in self.feature_map[(self.feature_map["ftype"] == "Label")][["name"]].values:
        target[name] = record.pop(name)
    return record, target
