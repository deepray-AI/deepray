# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import tensorflow as tf

from deepray.datasets.amazon_books_2014.defaults import (
  CARDINALITY_SELECTOR,
  NEGATIVE_HISTORY_CHANNEL,
  POSITIVE_HISTORY_CHANNEL,
  TARGET_ITEM_FEATURES_CHANNEL,
  USER_FEATURES_CHANNEL,
)
from deepray.layers.ctr_classification_mlp import CTRClassificationMLP

# from deepray.layers.embedding import Embedding


class EmbeddingInitializer(tf.keras.initializers.Initializer):
  def __call__(self, shape, dtype=tf.float32):
    maxval = tf.sqrt(tf.constant(1.0) / tf.cast(shape[0], tf.float32))
    maxval = tf.cast(maxval, dtype=dtype)
    minval = -maxval

    weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    weights = tf.cast(weights, dtype=tf.float32)
    return weights

  def get_config(self):
    return {}


# https://github.com/NVIDIA/DeepLearningExamples/blob/81ee705868a11d6fe18c12d237abe4a08aab5fd6/TensorFlow2/Recommendation/DLRM/embedding.py#L94
class Embedding(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim, *, trainable=True, embedding_name=None, initializer=EmbeddingInitializer()):
    super(Embedding, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embedding_name = embedding_name if embedding_name is not None else "embedding_table"
    self.embedding_table = None
    self.trainable = trainable
    self.initializer = initializer

  def build(self, input_shape):
    self.embedding_table = self.add_weight(
      self.embedding_name,
      shape=[self.input_dim, self.output_dim],
      dtype=tf.float32,
      initializer=self.initializer,
      trainable=self.trainable,
    )

  def call(self, indices):
    return tf.gather(params=self.embedding_table, indices=indices)


class SequentialRecommenderModel(tf.keras.Model, ABC):
  def __init__(self, feature_spec, embedding_dim, classifier_dense_sizes=(200,)):
    super(SequentialRecommenderModel, self).__init__()
    self.embedding_dim = embedding_dim

    features = feature_spec.feature_spec
    channel_spec = feature_spec.channel_spec

    embedding_names = []
    user_feature_fstring = "user_feat{}"
    item_feature_fstring = "item_feat{}"

    # Features in the same embedding group will share embedding table
    embedding_group_counter = 0
    feature_groups_cardinalities = []
    self.feature_name_to_embedding_group = {}

    for i, user_feature in enumerate(channel_spec[USER_FEATURES_CHANNEL]):
      self.feature_name_to_embedding_group[user_feature] = embedding_group_counter

      cardinality = features[user_feature][CARDINALITY_SELECTOR]
      feature_groups_cardinalities.append(cardinality)

      embedding_names.append(user_feature_fstring.format(i))

      embedding_group_counter += 1

    # Group corresponding item features from different item channels together
    zipped_item_features = zip(
      channel_spec[TARGET_ITEM_FEATURES_CHANNEL],
      channel_spec[POSITIVE_HISTORY_CHANNEL],
      channel_spec[NEGATIVE_HISTORY_CHANNEL],
    )

    for i, (feature_target, feature_pos, feature_neg) in enumerate(zipped_item_features):
      self.feature_name_to_embedding_group[feature_target] = embedding_group_counter
      self.feature_name_to_embedding_group[feature_pos] = embedding_group_counter
      self.feature_name_to_embedding_group[feature_neg] = embedding_group_counter

      cardinality = features[feature_target][CARDINALITY_SELECTOR]
      feature_groups_cardinalities.append(cardinality)

      embedding_names.append(item_feature_fstring.format(i))

      embedding_group_counter += 1

    self.variable_embeddings_groups = []
    for embedding_name, cardinality in zip(embedding_names, feature_groups_cardinalities):
      self.variable_embeddings_groups.append(
        Embedding(
          embedding_name=embedding_name,
          input_dim=cardinality + 1,  # ids in range <1, cardinality> (boundries included)
          output_dim=embedding_dim,
        )
      )

    self.classificationMLP = CTRClassificationMLP(layer_sizes=classifier_dense_sizes)

  def embed(self, features):
    embeddings = []
    for variable, id in features.items():
      embedding_group = self.feature_name_to_embedding_group[variable]

      embeddings.append(self.variable_embeddings_groups[embedding_group](id))
    return tf.concat(embeddings, -1)

  @abstractmethod
  def call(self, inputs):
    pass
