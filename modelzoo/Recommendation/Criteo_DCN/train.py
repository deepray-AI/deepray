import os
import sys

import tensorflow as tf
from absl import flags
from tf_keras.optimizers.legacy import Adam

import deepray as dp
from datasets.custom_dataset import CustomParquetPipeline
from dcn_v2 import Ranking
from deepray.callbacks import ModelCheckpoint
from deepray.callbacks.training_speed import TrainingSpeed
from deepray.core.trainer import Trainer
from deepray.utils import logging_util
from deepray.utils.export import export_to_savedmodel

logger = logging_util.get_logger()


def define_flags():
  flags.mark_flag_as_required('model_dir')
  flags.FLAGS(sys.argv)


def main():
  define_flags()
  pid = os.getpid()
  # input("pid: " + str(pid) +", press enter to continue")
  if flags.FLAGS.use_dynamic_embedding:
    from tensorflow_recommenders_addons import dynamic_embedding as de
    optimizer = Adam(learning_rate=flags.FLAGS.learning_rate, amsgrad=False)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=flags.FLAGS.use_horovod)
  else:
    optimizer = dp.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate, amsgrad=False)
    # optimizer = dp.optimizers.SGD(flags.FLAGS.learning_rate)
    # optimizer = dp.optimizers.Adagrad(learning_rate=flags.FLAGS.learning_rate)
    # optimizer = dp.optimizers.FtrlOptimizer(learning_rate=flags.FLAGS.learning_rate)
  model = Ranking(interaction="cross", use_group_embedding=False)

  data_pipe = CustomParquetPipeline()
  train_ds = data_pipe(
      batch_size=flags.FLAGS.batch_size,
      input_file_pattern=[
          "/workspaces/datasets/criteo-small-00000.parquet",
          "/workspaces/datasets/criteo-small-01799.parquet",
      ]
  )
  valid_ds = data_pipe(
      batch_size=flags.FLAGS.batch_size,
      input_file_pattern=[
          "/workspaces/datasets/criteo-small-01799.parquet",
      ]
  )

  trainer = Trainer(
      model=model,
      optimizer=optimizer,
      loss="binary_crossentropy",
      metrics=['AUC'],
      jit_compile=False,
      run_eagerly=flags.FLAGS.run_eagerly
  )
  # Create a TensorBoard callback
  logdir = os.path.join(flags.FLAGS.model_dir, 'tensorboard')
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch='5,52')
  trainer.fit(
      x=train_ds,
      epochs=flags.FLAGS.epochs,
      # verbose=0,
      # steps_per_epoch=460,
      # validation_data=valid_ds,
      # validation_steps=191/get_world_size()-1,
      callbacks=[
          tboard_callback,
          TrainingSpeed(),
          ModelCheckpoint(),
      ],
  )
  savedmodel_path = export_to_savedmodel(model)
  print(savedmodel_path)


if __name__ == "__main__":
  dp.runner(main)
