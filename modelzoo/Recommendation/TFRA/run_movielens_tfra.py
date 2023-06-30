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
from deepray.datasets.movielens import Movielens1MRating
from deepray.models.rec.dual_channels_deep_model import DualChannelsDeepModel

FLAGS = flags.FLAGS
FLAGS(
    [
        sys.argv[0],
        # "--distribution_strategy=off",
        # "--run_eagerly=false",
        "--steps_per_summary=20",
        "--use_dynamic_embedding=True",
    ]
)


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Movielens1MRating()
  with distribution_utils.get_strategy_scope(_strategy):
    model = DualChannelsDeepModel(
        user_embedding_size=32,
        movie_embedding_size=32,
        embedding_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5),
    )

  trainer = Trainer(
      model_or_fn=model,
      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
      # metrics=[tf.keras.metrics.AUC(num_thresholds=1000)],
      use_horovod=FLAGS.use_horovod,
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,
              # run_eagerly=False,
             )

  trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
