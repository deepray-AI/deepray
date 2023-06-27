from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app, flags

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.fashion_mnist import FashionMNIST

FLAGS = flags.FLAGS


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = FashionMNIST()
  with distribution_utils.get_strategy_scope(_strategy):
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  trainer = Trainer(
    model_or_fn=model,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(
    train_input=train_input_fn,
  )

  trainer.export_tfra()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
