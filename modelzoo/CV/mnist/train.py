from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import keras
import numpy as np
import tensorflow as tf
from absl import flags

import deepray as dp
from deepray.core.trainer import Trainer
from deepray.datasets.mnist import Mnist


def define_flasg():
  flags.FLAGS([
    sys.argv[0],
    "--train_data=mnist",
    # "--run_eagerly=true",
    "--steps_per_execution=1",
    # "--batch_size=1024",
  ])


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


def main():
  mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
  ])

  trainer = Trainer(
    optimizer=tf.keras.optimizers.Adam(0.001),
    model=mnist_model,
    loss=tf.losses.SparseCategoricalCrossentropy(),
    # loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
  )
  data_pipe = Mnist()
  train_input = data_pipe(flags.FLAGS.batch_size, is_training=True)
  test_input = data_pipe(flags.FLAGS.batch_size, is_training=False)
  tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(flags.FLAGS.model_dir, "tensorboard"), histogram_freq=1, profile_batch="1,2"
  )

  trainer.fit(
    train_input=train_input,
    eval_input=test_input,
    callbacks=[tboard_callback, EarlyStoppingAtMinLoss()],
  )


if __name__ == "__main__":
  dp.runner(main)
