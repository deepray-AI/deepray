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

import tensorflow as tf
from absl import app, flags

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.movielens import Movielens100kRating
from demo import Demo

FLAGS = flags.FLAGS


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Movielens100kRating()
  with distribution_utils.get_strategy_scope(_strategy):
    model = Demo(embedding_size=32)

  trainer = Trainer(
      model_or_fn=model,
      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,)

  # trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
