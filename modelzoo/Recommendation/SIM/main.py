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
from deepray.models.rec.sim_model import SIMModel
from deepray.models.rec.din_model import DINModel
from deepray.models.rec.dien_model import DIENModel

FLAGS = flags.FLAGS
FLAGS(
    [
        sys.argv[0],
        "--train_data=movielens/100k-ratings",
        # "--distribution_strategy=off",
        # "--run_eagerly=true",
        "--steps_per_summary=20",
        "--use_dynamic_embedding=True",
        # "--batch_size=1024",
    ]
)


def build_sim_loss_fn(alpha=1.0, beta=1.0):
  cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  @tf.function
  def sim_loss_fn(targets, gsu_logits, esu_logits):
    gsu_loss = cross_entropy_loss(targets, gsu_logits)
    esu_loss = cross_entropy_loss(targets, esu_logits)
    return 0.5 * (alpha * gsu_loss + beta * esu_loss)

  return sim_loss_fn


@tf.function
def dien_auxiliary_loss_fn(click_probs, noclick_probs, mask=None):
  if mask is None:
    mask = tf.ones_like(click_probs)
  click_loss_term = -tf.math.log(click_probs) * mask
  noclick_loss_term = -tf.math.log(1.0 - noclick_probs) * mask

  return tf.reduce_mean(click_loss_term + noclick_loss_term)


def build_model_and_loss(model_params):
  if FLAGS.model_type == "sim":
    model = SIMModel(
        model_params['feature_spec'],
        mlp_hidden_dims=model_params["mlp_hidden_dims"],
        embedding_dim=model_params["embedding_dim"],
        dropout_rate=model_params["dropout_rate"]
    )
    classification_loss_fn = build_sim_loss_fn()

    def classification_loss(targets, output_dict):
      """ compute loss."""
      return classification_loss_fn(targets, output_dict["stage_one_logits"], output_dict["stage_two_logits"])

    dien_aux_loss = dien_auxiliary_loss_fn(
        output_dict["aux_click_probs"],
        output_dict["aux_noclick_probs"],
        mask=mask_for_aux_loss,
    )

    total_loss = classification_loss + dien_aux_loss
    loss = {"total_loss": total_loss, "classification_loss": classification_loss, "dien_aux_loss": dien_aux_loss}

    @tf.function
    def model_fn(batch, training=True):
      input_data, targets = batch
      # take the mask for N-1 timesteps from prepared input data
      mask_for_aux_loss = input_data["short_sequence_mask"][:, 1:]

      # model forward pass
      output_dict = model(input_data, training=training)

      logits = output_dict["stage_two_logits"]

      loss_dict = {"total_loss": total_loss, "classification_loss": classification_loss, "dien_aux_loss": dien_aux_loss}

      return (targets, logits), loss_dict
  elif FLAGS.model_type == "dien":
    model = DIENModel(
        model_params['feature_spec'],
        mlp_hidden_dims={
            "classifier": model_params["mlp_hidden_dims"]["stage_2"],
            "aux": model_params["mlp_hidden_dims"]["aux"],
        },
        embedding_dim=model_params["embedding_dim"],
    )
    classification_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    class CustomLossClass:

      def __call__(self, targets, output_dict):
        classification_loss = classification_loss_fn(targets, output_dict["logits"])

        dien_aux_loss = dien_auxiliary_loss_fn(
            output_dict["aux_click_probs"],
            output_dict["aux_noclick_probs"],
            mask=input_data["short_sequence_mask"][:, 1:],
        )

        total_loss = classification_loss + dien_aux_loss

    @tf.function
    def model_fn(batch, training=True):
      input_data, targets = batch
      # take the mask for N-1 timesteps from prepared input data
      mask_for_aux_loss = input_data["short_sequence_mask"][:, 1:]

      # model forward pass
      output_dict = model(input_data, training=training)

      # compute loss
      classification_loss = classification_loss_fn(targets, output_dict["logits"])

      dien_aux_loss = dien_auxiliary_loss_fn(
          output_dict["aux_click_probs"],
          output_dict["aux_noclick_probs"],
          mask=mask_for_aux_loss,
      )

      total_loss = classification_loss + dien_aux_loss

      logits = output_dict["logits"]

      loss_dict = {"total_loss": total_loss, "classification_loss": classification_loss, "dien_aux_loss": dien_aux_loss}

      return (targets, logits), loss_dict
  elif FLAGS.model_type == "din":
    model = DINModel(
        model_params['feature_spec'],
        mlp_hidden_dims=model_params["mlp_hidden_dims"]["stage_2"],
        embedding_dim=model_params["embedding_dim"]
    )
    classification_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def model_fn(batch, training=True):
      input_data, targets = batch

      # model forward pass
      output_dict = model(input_data, training=training)

      # compute loss
      total_loss = classification_loss_fn(targets, output_dict["logits"])

      logits = output_dict["logits"]

      loss_dict = {"total_loss": total_loss}

      return (targets, logits), loss_dict

  return model, model_fn


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Movielens100kRating()
  with distribution_utils.get_strategy_scope(_strategy):
    model = build_model_and_loss()

  trainer = Trainer(
      model=model,
      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,)

  trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
