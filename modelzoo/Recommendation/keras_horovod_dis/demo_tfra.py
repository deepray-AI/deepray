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

import tensorflow as tf
from absl import app, flags

from deepray.core.trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.movielens import Movielens100kRating
from deepray.models.rec.tfra_demo import build_keras_model

FLAGS(
    [
        sys.argv[0],
        "--train_data=movielens/100k-ratings",
        # "--distribution_strategy=off",
        # "--run_eagerly=true",
        "--steps_per_execution=20",
        "--use_dynamic_embedding=True",
        # "--batch_size=1024",
    ]
)


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Movielens100kRating()
  with distribution_utils.get_strategy_scope(_strategy):
    import horovod.tensorflow as hvd
    model = build_keras_model(is_training=True, mpi_size=hvd.size(), mpi_rank=hvd.rank())

  trainer = Trainer(
      model=model,
      loss="binary_crossentropy",
      metrics=tf.keras.metrics.AUC(num_thresholds=1000, summation_method='minoring'),
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,)

  # trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
