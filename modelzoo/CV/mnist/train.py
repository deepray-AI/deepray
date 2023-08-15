from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
from absl import app, flags
from datetime import datetime

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.mnist import Mnist

FLAGS = flags.FLAGS
FLAGS(
    [
        sys.argv[0],
        "--train_data=mnist",
        # "--distribution_strategy=off",
        # "--run_eagerly=true",
        "--steps_per_summary=10",
        # "--use_horovod=True",
        # "--batch_size=1024",
    ]
)


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = Mnist()
  with distribution_utils.get_strategy_scope(_strategy):
    mnist_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

  trainer = Trainer(
      optimizer=tf.keras.optimizers.Adam(0.001),
      model_or_fn=mnist_model,
      loss=tf.losses.SparseCategoricalCrossentropy(),
      # loss='sparse_categorical_crossentropy',
      metrics=["accuracy"]
  )

  tboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(FLAGS.model_dir, 'tensorboard'), histogram_freq=1, profile_batch='10,20'
  )

  train_input = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input, callbacks=[tboard_callback])


if __name__ == "__main__":
  app.run(main)
