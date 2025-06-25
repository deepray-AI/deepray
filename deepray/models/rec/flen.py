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

import tensorflow as tf
from absl import flags
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

import deepray as dp
from deepray.layers.embedding import Embedding
from deepray.utils.data.feature_map import FeatureMap
from deepray.layers.field_wise_bi_interaction import FieldWiseBiInteraction

__all__ = [
  "FLEN",
]


class FLEN(Model):
  def __init__(self, field_info, embedding_dim=16):
    if not field_info or not isinstance(field_info, dict):
      raise ValueError("Must specify field_info")

    super(FLEN, self).__init__(name="FLEN")

    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map

    self.embedding_layers = {}
    self.field_info = field_info

    for name, ftype, dtype, voc_size, length in self.feature_map[(self.feature_map["ftype"] == "Categorical")][
      ["name", "ftype", "dtype", "voc_size", "length"]
    ].values:
      self.embedding_layers[name] = Embedding(
        embedding_dim=embedding_dim, vocabulary_size=voc_size + 1, name="embedding_" + name
      )

    # 2. mlp part
    self.deep_fc_64 = dp.layers.FullyConnect(units=64, activation="relu")
    self.deep_bn_64 = BatchNormalization(momentum=0.9)
    self.deep_dropout_64 = Dropout(rate=0.2)
    self.deep_fc_32 = dp.layers.FullyConnect(units=32, activation="relu")
    self.deep_bn_32 = BatchNormalization(momentum=0.9)
    self.deep_dropout_32 = Dropout(rate=0.2)

    # 3. field-weighted embedding
    self.fwbi = FieldWiseBiInteraction(num_fields=len(field_info.keys()), embedding_size=embedding_dim)
    self.fwbi_fc_32 = dp.layers.FullyConnect(units=32, activation="relu")
    self.fwbi_bn = BatchNormalization(momentum=0.9)
    self.fwbi_drop = Dropout(rate=0.2)

    self.logits = dp.layers.FullyConnect(units=1, activation="sigmoid")

  def call(self, inputs, training=False):
    embedding = {}
    for name, tensor in inputs.items():
      embedding[name] = self.embedding_layers[name](tensor)

    # 1. embedding field-wise features
    embeddings = list(embedding.values())
    embeddings = tf.concat(values=embeddings, axis=-1)
    # 2. deep mlp part
    deep_fc_64 = self.deep_fc_64(embeddings)
    deep_bn_64 = self.deep_bn_64(deep_fc_64)
    deep_dropout_64 = self.deep_dropout_64(deep_bn_64)
    deep_fc_32 = self.deep_fc_32(deep_dropout_64)
    deep_bn_32 = self.deep_bn_32(deep_fc_32)
    deep_dropout_32 = self.deep_dropout_32(deep_bn_32)

    # 3. field-weighted embedding
    field_embedding = [
      tf.reduce_mean(tf.stack([embedding[name] for name in names], axis=-1), axis=-1)
      for field, names in self.field_info.items()
    ]
    fwbi_ebm = tf.concat(field_embedding, axis=-1)

    fwbi = self.fwbi(fwbi_ebm)
    fwbi_fc_32 = self.fwbi_fc_32(fwbi)
    fwbi_bn = self.fwbi_bn(fwbi_fc_32)
    fwbi_drop = self.fwbi_drop(fwbi_bn)

    logits = tf.concat(values=[deep_dropout_32, fwbi_drop], axis=1)
    logits = self.logits(logits)

    return logits
