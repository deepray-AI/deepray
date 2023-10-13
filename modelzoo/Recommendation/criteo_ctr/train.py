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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app
from absl import flags
from tensorflow_recommenders_addons import dynamic_embedding as de

from dcn_v2 import Ranking
from deepray.core.base_trainer import Trainer
from deepray.datasets.criteo import CriteoTsvReader
from deepray.utils.export import export_to_savedmodel

FLAGS = flags.FLAGS


def main(_):
  model = Ranking(interaction="cross")

  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=False)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=FLAGS.use_horovod)

  trainer = Trainer(model=model, optimizer=optimizer, loss="binary_crossentropy", metrics=[
      'AUC',
  ])
  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn, steps_per_epoch=FLAGS.steps_per_epoch)

  import numpy as np
  a = {
      "feature_14":
          tf.constant(np.array([6394203, 7535249, 3500077, 836339, 7401745, 375123]), dtype=tf.int32),
      "feature_15":
          tf.constant(np.array([6394203, 7535249, 3500077, 836339, 7401745, 375123]), dtype=tf.int32),
      "dense_features":
          tf.constant(
              np.array(
                  [
                      [0.7361634, 0.7361634], [0.00337589, 0.00337589], [0.673707, 0.673707], [0.33169293, 0.33169293],
                      [0.8020003, 0.8020003], [0.18556607, 0.18556607]
                  ]
              ),
              dtype=tf.float32
          )
  }
  print(trainer.model(a))

  export_to_savedmodel(trainer.model)


if __name__ == "__main__":
  app.run(main)
