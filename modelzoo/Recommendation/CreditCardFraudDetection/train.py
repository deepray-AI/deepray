from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import flags
from tensorflow import keras

from deepray.core.trainer import Trainer
from deepray.datasets.creditcardfraud import CreditCardFraud
from tf_keras.optimizers.legacy import Adam

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def main():
  data_pipe = CreditCardFraud()
  train_input = data_pipe(flags.FLAGS.batch_size, is_training=True)
  output_bias = None
  input_dim = data_pipe.train_features.shape[-1]
  model = keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(input_dim,)),
          keras.layers.Dense(16, activation='relu'),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
      ]
  )

  optimizer = Adam(learning_rate=flags.FLAGS.learning_rate, amsgrad=False)
  trainer = Trainer(model=model, optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
  trainer.fit(x=train_input)


if __name__ == "__main__":
  main()
