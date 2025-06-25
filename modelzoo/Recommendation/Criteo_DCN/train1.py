import os
import sys

import tensorflow as tf
from absl import flags

import deepray as dp
from datasets.custom_dataset import CustomParquetPipeline
from dcn_v2 import Ranking
from deepray.callbacks import ModelCheckpoint
from deepray.core.trainer import Trainer
from deepray.utils import logging_util
from deepray.utils.export import export_to_savedmodel
from deepray.utils.horovod_utils import get_world_size
from tf_keras import optimizers

logger = logging_util.get_logger()


def define_flags():
  flags.mark_flag_as_required('model_dir')
  flags.FLAGS(sys.argv)


def main():
  define_flags()
  pid = os.getpid()
  # 验证设置
  print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
  print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
  # input("pid: " + str(pid) +", press enter to continue")
  if flags.FLAGS.use_dynamic_embedding:
    from tensorflow_recommenders_addons import dynamic_embedding as de
    optimizer = optimizers.legacy.Adam(learning_rate=flags.FLAGS.learning_rate)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=flags.FLAGS.use_horovod)
  else:
    optimizer = dp.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate)
    # optimizer = dp.optimizers.SGD(flags.FLAGS.learning_rate)
  # optimizer = keras.optimizers.legacy.Adam(learning_rate=flags.FLAGS.learning_rate)
  # optimizer = dp.optimizers.Adagrad(learning_rate=flags.FLAGS.learning_rate)
  # optimizer = dp.optimizers.FtrlOptimizer(learning_rate=flags.FLAGS.learning_rate)
  model = Ranking(interaction="cross")
  # dense_opt = keras.optimizers.legacy.Adam(learning_rate=flags.FLAGS.learning_rate)
  # optimizer = MultiOptimizer([
  #     (embedding_opt, "DynamicVariable_"),
  # ], default_optimizer=dense_opt)
  data_pipe = CustomParquetPipeline()
  train_ds = data_pipe(
      batch_size=flags.FLAGS.batch_size,
      input_file_pattern=[
          "/workspaces/datasets/00000-1-038360cf-9d9d-454c-8381-6a57bdbf6d57-00001.parquet",
      ]
  )
  valid_ds = data_pipe(
      batch_size=flags.FLAGS.batch_size,
      input_file_pattern=[
          "/workspaces/datasets/01799-1800-26382079-2024-439e-84bf-e7b2231e0a2f-00001.parquet",
      ]
  )

  optimizer.global_step = model._train_counter
  model.compile(
      optimizer=optimizer,
      loss="binary_crossentropy",
      metrics=['AUC'],
      jit_compile=False,
      run_eagerly=flags.FLAGS.run_eagerly
  )
  # Create a TensorBoard callback
  logdir = os.path.join(flags.FLAGS.model_dir, 'tensorboard')
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch='5,52')
  # breakpoint()
  model.fit(
      x=train_ds,
      epochs=flags.FLAGS.epochs,
      steps_per_epoch=1000,
      validation_data=valid_ds,
      # validation_steps=191/get_world_size()-1,
      callbacks=[
          # tboard_callback
          # ModelCheckpoint(),
      ],
  )
  # savedmodel_path = export_to_savedmodel(model)
  # print(savedmodel_path)


if __name__ == "__main__":
  # dp.runner(main)
  main()
