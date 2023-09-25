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
from deepray.datasets.criteo import CriteoTsvReader
from dcn_v2 import Ranking
from tensorflow_recommenders_addons import dynamic_embedding as de

FLAGS = flags.FLAGS


def main(_):
  model = Ranking(interaction="cross")
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=False)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=FLAGS.use_horovod)

  trainer = Trainer(
    model_or_fn=model,
    optimizer=optimizer,
    loss="binary_crossentropy",
  )
  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn, )


if __name__ == "__main__":
  app.run(main)
