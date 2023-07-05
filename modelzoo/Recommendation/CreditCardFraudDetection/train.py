from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app, flags
from tensorflow import keras

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.creditcardfraud import CreditCardFraud

FLAGS = flags.FLAGS

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


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = CreditCardFraud()
  train_input = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  with distribution_utils.get_strategy_scope(_strategy):
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

  trainer = Trainer(model_or_fn=model, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)

  trainer.fit(train_input=train_input,)

  # trainer.export_tfra()


if __name__ == "__main__":
  app.run(main)
