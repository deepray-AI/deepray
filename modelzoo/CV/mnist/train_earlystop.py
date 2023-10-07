from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import keras
import numpy as np
import tensorflow as tf
from absl import app, flags

from deepray.core.base_trainer import Trainer
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


def get_model():
  return tf.keras.Sequential(
      [
          tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(1),
      ]
  )


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

  def __init__(self, patience=0):
    super().__init__()
    self.patience = patience
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  # def on_batch_begin(self, batch, logs=None):
  #   pass

  # def on_batch_end(self, batch, logs=None):
  #   if batch < 5:
  #     print(batch, self.model.get_weights()[0][0][0][0])
  #   pass

  def on_epoch_end(self, epoch, logs=None):
    print(logs)
    current = logs.get("loss")
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print("Restoring model weights from the end of the best epoch.")
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def main(_):
  data_pipe = Mnist()
  model = get_model()

  trainer = Trainer(
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
      model=model,
      loss="mean_squared_error",
      metrics=["mean_absolute_error"],
  )

  callbacks = [EarlyStoppingAtMinLoss()],

  train_input = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(
      train_input=train_input,
      callbacks=callbacks,
  )


if __name__ == "__main__":
  app.run(main)
