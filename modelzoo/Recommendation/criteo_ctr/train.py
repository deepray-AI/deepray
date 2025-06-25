import os
import sys

import tensorflow as tf
from absl import flags

import deepray as dp
from datasets.custom_dataset import CustomArsenalParquetDataset
from dcn_v2 import Ranking
from deepray.callbacks import ModelCheckpoint
from deepray.callbacks.training_speed import TrainingSpeed
from deepray.core.trainer import Trainer
from deepray.utils import logging_util
from deepray.utils.export import export_to_savedmodel

from deepray.utils.horovod_utils import get_world_size, is_main_process

logger = logging_util.get_logger()


def define_flags():
  flags.mark_flag_as_required("model_dir")
  flags.FLAGS(sys.argv)


def build_dataset(split="train", version="criteo-small"):
  data_pipe = CustomArsenalParquetDataset(
    dataset_name=flags.FLAGS.dataset, partitions=[{"version": version, "split": split}]
  )
  if split == "valid":
    is_training = False
  else:
    is_training = True

  file_list = data_pipe.get_dataset_files()
  dataset = data_pipe(
    input_file_pattern=file_list,
    batch_size=flags.FLAGS.batch_size,
    is_training=is_training,
    shuffle=True,
    shuffle_buffer=20,
  )
  steps = data_pipe.get_hvd_step(flags.FLAGS.batch_size, file_list=file_list)
  logger.info(f"steps = {steps}")
  return dataset, steps


def main():
  define_flags()
  pid = os.getpid()
  # 验证设置
  print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
  print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
  # input("pid: " + str(pid) +", press enter to continue")
  if flags.FLAGS.use_dynamic_embedding:
    from tensorflow_recommenders_addons import dynamic_embedding as de

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=flags.FLAGS.learning_rate)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=flags.FLAGS.use_horovod)
  else:
    # optimizer = dp.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate)
    optimizer = dp.optimizers.SGD(flags.FLAGS.learning_rate)
    # optimizer = dp.optimizers.Adagrad(learning_rate=flags.FLAGS.learning_rate)
    # optimizer = dp.optimizers.FtrlOptimizer(learning_rate=flags.FLAGS.learning_rate)
  model = Ranking(interaction="cross", use_group_embedding=True)

  train_ds, train_steps = build_dataset("train")
  # valid_ds, valid_steps = build_dataset("validation")
  # test_ds, test_steps = build_dataset("test")
  trainer = Trainer(model=model, optimizer=optimizer, loss="binary_crossentropy", metrics=["AUC"], jit_compile=False)
  # Create a TensorBoard callback
  logdir = os.path.join(flags.FLAGS.model_dir, "tensorboard")
  # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch='5,52')
  trainer.fit(
    x=train_ds,
    steps_per_epoch=train_steps,
    # eval_input=valid_ds,
    # eval_steps=valid_steps,
    callbacks=[
      TrainingSpeed(),
      # tboard_callback,
      # ModelCheckpoint(),
    ],
  )
  savedmodel_path = export_to_savedmodel(model)
  print(savedmodel_path)


if __name__ == "__main__":
  dp.runner(main)
