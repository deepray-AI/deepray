import sys

import tensorflow as tf
from absl import flags

import deepray as dp
from dcn_v2 import Ranking
from deepray.core.trainer import Trainer
from deepray.datasets.criteo import CriteoTsvReader
from deepray.utils import logging_util

logger = logging_util.get_logger()


def define_flags():
  argv = sys.argv + [
      "--feature_map=feature_map_small.csv",
      "--init_weights=/code/results/tf_tfra_training_criteo_dcn_fp32_gbs1024_240102150028/export_main/variables/",
  ]
  flags.FLAGS(argv)


def main():
  define_flags()
  model = Ranking(interaction="cross")

  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  train_ds = data_pipe(flags.FLAGS.train_data, flags.FLAGS.batch_size, is_training=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate, amsgrad=False)
  if flags.FLAGS.use_dynamic_embedding:
    from tensorflow_recommenders_addons import dynamic_embedding as de
    optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=flags.FLAGS.use_horovod)

  trainer = Trainer(model=model, optimizer=optimizer, loss="binary_crossentropy", metrics=['AUC'])
  trainer.evaluate(eval_input=train_ds, eval_steps=100)
  # model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['AUC'])
  # model.load_weights("/code/results/tf_tfra_training_criteo_dcn_fp32_gbs1024_240102150028/export_main/variables/variables").expect_partial()
  # model.evaluate(train_ds, steps=100, return_dict=True)


if __name__ == "__main__":
  dp.runner(main)
