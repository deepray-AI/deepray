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
"""Run dien on AmazonBooks2014 in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app, flags

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.amazon_books_2014 import AmazonBooks2014
from deepray.models.rec.sim_model import SIMModel
from .feature_spec import FeatureSpec

FLAGS = flags.FLAGS


def build_sim_loss_fn(alpha=1.0, beta=1.0):
  cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  @tf.function
  def sim_loss_fn(targets, gsu_logits, esu_logits):
    gsu_loss = cross_entropy_loss(targets, gsu_logits)
    esu_loss = cross_entropy_loss(targets, esu_logits)
    return 0.5 * (alpha * gsu_loss + beta * esu_loss)

  return sim_loss_fn


def custom_loss_fn(y_true, y_pred):
  return y_pred


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  feature_spec = FeatureSpec.from_yaml(FLAGS.config_file)
  data_pipe = AmazonBooks2014(FLAGS.max_seq_length)
  with distribution_utils.get_strategy_scope(_strategy):
    model_params = {
        "feature_spec": feature_spec,
        "embedding_dim": FLAGS.embedding_dim,
        "mlp_hidden_dims":
            {
                "stage_1": FLAGS.stage_one_mlp_dims,
                "stage_2": FLAGS.stage_two_mlp_dims,
                "aux": FLAGS.aux_mlp_dims
            },
        "dropout_rate": FLAGS.dropout_rate,
        # "model_type": model_type
    }

    model = SIMModel(
        model_params['feature_spec'],
        mlp_hidden_dims=model_params["mlp_hidden_dims"],
        embedding_dim=model_params["embedding_dim"],
        dropout_rate=model_params["dropout_rate"]
    )

  trainer = Trainer(
      model_or_fn=model,
      loss={
          "stage_one_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
          "stage_two_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
          "auxiliary_logits": custom_loss_fn
      },
      loss_weights={
          "stage_one_logits": 0.5,
          "stage_two_logits": 0.5,
          "auxiliary_logits": 1
      },
      metrics={"stage_two_logits": tf.keras.metrics.AUC(num_thresholds=8000, name="auc_accumulator", from_logits=True)},
  )

  # since each tfrecord file must include all of the features, it is enough to read first chunk for each split.
  train_dataset = data_pipe(FLAGS.train_data, batch_size=FLAGS.batch_size, prebatch_size=FLAGS.prebatch)
  trainer.fit(train_input=train_dataset,)


if __name__ == "__main__":
  app.run(main)
