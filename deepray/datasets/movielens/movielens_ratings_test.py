#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import sys, os
import deepray as dp
from datetime import datetime
from absl import app, flags, logging

from deepray.datasets.movielens.movielens_100k_ratings import Movielens100kRating

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def define_flags():
  argv = sys.argv + [
    "--epochs=1",
    "--batch_size=2",
    "--train_data=movielens/100k-ratings",
  ]
  flags.FLAGS(argv)


def runner():
  data_pipe = Movielens100kRating()
  train_dataset = data_pipe(flags.FLAGS.batch_size, is_training=True)
  num_examples = 0

  for x in train_dataset:
    num_examples += flags.FLAGS.batch_size

  print(x)
  print(num_examples)


if __name__ == "__main__":
  dp.runner(runner)
