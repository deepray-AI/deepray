import tensorflow as tf
from absl import flags
import datetime, os
import deepray as dp
from deepray.core.trainer import Trainer
from deepray.datasets.fashion_mnist import FashionMNIST


def main():
  model = tf.keras.models.Sequential(
      [
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation='softmax')
      ]
  )

  trainer = Trainer(
      model=model,
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  data_pipe = FashionMNIST()
  train_input_fn = data_pipe(flags.FLAGS.batch_size, is_training=True)

  # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  logdir = os.path.join(flags.FLAGS.model_dir, 'tensorboard')

  tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
  trainer.fit(train_input=train_input_fn, callbacks=[tensorboard_callback])


if __name__ == "__main__":
  dp.runner(main)
