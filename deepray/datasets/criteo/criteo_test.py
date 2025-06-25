#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys
from datetime import datetime
from deepray.utils.flags import core as flags_core

from absl import app, flags, logging

from deepray.datasets.criteo.criteo import Criteo

flags_core.define_base(
  model_dir=False,
  clean=True,
  epochs=True,
  epochs_between_evals=False,
  export_dir=False,
  stop_threshold=False,
)

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
      sys.argv[0],
      "--batch_size=16",
      "-epochs=1",
      # '--train_data=hdfs://10.11.11.241:8020/test/offline_feature_model_topic_parquet/ymd=2021111[0-6]/*.parquet',
      "--train_data=/Users/admin/Downloads/train.csv",
      f"--feature_map={dir_path}/feature_map.csv",
      "--prefetch_buffer=64",
      "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)
  data_pipe = Criteo()
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size)
  for x in train_dataset:
    print(x)

  # test_dataset = data_pipe(test_file, 4, 1)


if __name__ == "__main__":
  app.run(runner)
