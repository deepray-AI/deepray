#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import json
import os
import sys
from datetime import datetime

import tensorflow as tf
from absl import app, flags

from .openwebtext import Openwebtext

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  if len(argv) <= 1:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    argv = [
      sys.argv[0],
      "--batch_size=10240",
      "-epochs=1",
      f"--train_data=/workspaces/dataset/openwebtext/pretrain_tfrecords/*",
      # "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = Openwebtext(max_seq_length=128)
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  num_examples = 0
  for x in train_dataset:
    num_examples += FLAGS.batch_size
    if num_examples % 100 == 0:
      print(num_examples)

  print(x)
  print(num_examples)


if __name__ == "__main__":
  app.run(runner)
