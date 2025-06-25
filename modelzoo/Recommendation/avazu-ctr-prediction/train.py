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

from absl import flags

import tensorflow as tf
from absl import app
from absl import flags
import tensorflow_recommenders_addons as tfra

from deepray.core.trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.avazu import Avazu


def main(_):
  field_info = {
      "user": [
          'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C1', 'device_model', 'device_type', 'device_id'
      ],
      "context": [
          'banner_pos',
          'site_id',
          'site_domain',
          'site_category',
          'device_conn_type',
          'hour',
      ],
      "item": [
          'app_id',
          'app_domain',
          'app_category',
      ]
  }

  if flags.FLAGS.model_name == "flen":
    from deepray.models.rec.flen import FLEN as mymodel
  elif flags.FLAGS.model_name == "flend":
    from deepray.models.rec.flend import FLEND as mymodel
  elif flags.FLAGS.model_name == "ccpm":
    from .ccpm_diamond import CCPM as mymodel

  _strategy = distribution_utils.get_distribution_strategy()
  with distribution_utils.get_strategy_scope(_strategy):
    model = mymodel(field_info=field_info, embedding_dim=16)
    optimizer = tf.keras.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate)
    if flags.FLAGS.use_dynamic_embedding:
      optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

  data_pipe = Avazu()
  train_dataset = data_pipe(flags.FLAGS.train_data, flags.FLAGS.batch_size, is_training=True)
  valid_g2b = data_pipe(flags.FLAGS.valid_data, flags.FLAGS.batch_size, is_training=False)

  trainer = Trainer(
      model=model,
      loss=tf.keras.losses.BinaryCrossentropy(),
      # optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.03, initial_accumulator_value=1e-3),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.AUC()]
  )

  trainer.fit(
      x=train_dataset,
      validation_data=valid_g2b,
      # callbacks=[
      #   # Write TensorBoard logs to `./logs` directory
      #   tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir, histogram_freq=1, profile_batch=3),
      #   # tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir),
      # ]
  )


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
