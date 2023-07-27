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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app, flags
import tensorflow as tf
from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.movielens import Movielens100kRating
from tensorflow.keras.layers import Dense

from deepray.layers.embedding import Embedding
from deepray.utils.data.feature_map import FeatureMap

FLAGS = flags.FLAGS
FLAGS(
    [
        sys.argv[0],
        "--train_data=movielens/100k-ratings",
        # "--distribution_strategy=off",
        # "--run_eagerly=true",
        "--steps_per_summary=20",
        "--feature_map=/workspaces/deepray/deepray/datasets/movielens/movielens.csv",
        # "--batch_size=1024",
    ]
)


class Demo(tf.keras.Model):

  def __init__(self, embedding_size, is_training=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if is_training:
      initializer = tf.keras.initializers.VarianceScaling()
    else:
      initializer = tf.keras.initializers.Zeros()
    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map
    self.features_dict = {}
    for key, dtype, emb_size, length in self.feature_map.loc[self.feature_map["ftype"] == "Categorical"][[
        "name", "dtype", "dim", "length"
    ]].values:
      self.features_dict[key] = Embedding(
          vocabulary_size=100,
          embedding_dim=emb_size,
          # key_dtype=dtype,
          # value_dtype=tf.float32,
          # initializer=initializer,
          name=key + '_DynamicEmbeddingLayer',
      )

    self.d0 = Dense(
        256,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )
    self.d1 = Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )
    self.d2 = Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )

  # @tf.function
  def call(self, features, *args, **kwargs):

    movie_id = features["movie_id"]
    user_id = features["user_id"]

    user_id_weights = self.features_dict['user_id'](user_id)

    movie_id_weights = self.features_dict['movie_id'](movie_id)

    embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)

    dnn = self.d0(embeddings)
    dnn = self.d1(dnn)
    dnn = self.d2(dnn)
    out = tf.reshape(dnn, shape=[-1])

    return out
    # loss = tf.keras.losses.MeanSquaredError()(rating, out)
    # predictions = {"out": out}


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Movielens100kRating()
  with distribution_utils.get_strategy_scope(_strategy):
    model = Demo(embedding_size=32)

  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=True)

  trainer = Trainer(
      optimizer=optimizer,
      model_or_fn=model,
      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,)

  # trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
