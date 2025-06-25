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
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

import deepray as dp

__all__ = [
  "FLEND",
]


class FLEND(Model):
  def __init__(self, field_info, embedding_dim=16):
    if not field_info or not isinstance(field_info, dict):
      raise ValueError("Must specify field_info")

    super(FLEND, self).__init__(name="FLEND")

    self.embedding_layers = {}
    self.dice_bn_layer = BatchNormalization(momentum=0.9)
    self.dice_dropout_layer = Dropout(0.7)
    self.dice_fc_layers = dp.layers.FullyConnect(units=32, name="dice_fc")
    # self.sparse_inputs = {}
    for k, v in field_info.items():
      self.embedding_layers[k] = dp.layers.FullyConnectv2(units=32, input_shape=(v,), name="fc_" + k)
      # self.sparse_inputs[k] = tf.keras.Input(name=k, shape=(v,), sparse=True)

    # 2. mlp part
    self.deep_fc_64 = dp.layers.FullyConnect(units=64, activation="relu")
    self.deep_bn_64 = BatchNormalization(momentum=0.9)
    self.deep_dropout_64 = Dropout(rate=0.2)
    self.deep_fc_32 = dp.layers.FullyConnect(units=32, activation="relu")
    self.deep_bn_32 = BatchNormalization(momentum=0.9)
    self.deep_dropout_32 = Dropout(rate=0.2)

    # 3. field-weighted embedding
    self.fwbi = dp.layers.FieldWiseBiInteraction(num_fields=3, embedding_size=32)
    self.fwbi_fc_32 = dp.layers.FullyConnect(units=32, activation="relu")
    self.fwbi_bn = BatchNormalization(momentum=0.9)
    self.fwbi_drop = Dropout(rate=0.2)

    self.fwbi_bn2 = BatchNormalization(momentum=0.9)

    self.logits = dp.layers.FullyConnect(units=1, activation="sigmoid")
    # self._set_inputs(self.sparse_inputs)

  def call(self, inputs, training=False):
    # 1. embedding field-wise features
    embeddings = []
    fm_embeds = []
    for k, v in inputs.items():
      embedding, fm_embed = self.embedding_layers[k](v)
      embeddings.append(embedding)
      fm_embeds.append(fm_embed)

    embeddings = tf.concat(values=embeddings, axis=1)
    fm_embedding = tf.concat(fm_embeds, axis=1)
    fm_embedding = self.dice_fc_layers(fm_embedding)
    fm_embedding = self.dice_bn_layer(fm_embedding)
    fm_embedding = self.dice_dropout_layer(fm_embedding)

    # 2. deep mlp part
    deep_fc_64 = self.deep_fc_64(embeddings)
    deep_bn_64 = self.deep_bn_64(deep_fc_64)
    deep_dropout_64 = self.deep_dropout_64(deep_bn_64)
    deep_fc_32 = self.deep_fc_32(deep_dropout_64)
    deep_bn_32 = self.deep_bn_32(deep_fc_32)
    deep_dropout_32 = self.deep_dropout_32(deep_bn_32)

    # 3. field-weighted embedding
    fwbi = self.fwbi(embeddings)
    fwbi_fc_32 = self.fwbi_fc_32(fwbi)
    fwbi_bn = self.fwbi_bn(fwbi_fc_32)
    fwbi_drop = self.fwbi_drop(fwbi_bn)

    fwbi = tf.add(fwbi_drop, fm_embedding)
    fwbi = self.fwbi_bn2(fwbi)

    logits = tf.concat(values=[deep_dropout_32, fwbi], axis=1)
    logits = self.logits(logits)

    return logits
