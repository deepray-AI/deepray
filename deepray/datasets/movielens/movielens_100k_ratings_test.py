#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import sys
from datetime import datetime

from absl import app, flags, logging

from deepray.datasets.movielens.movielens_100k_ratings import Movielens100kRating

FLAGS = flags.FLAGS
logging.set_verbosity(logging.INFO)

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        # "--batch_size=16",
        "-epochs=1",
        "--train_data=movielens/100k-ratings",
        # f"--feature_map={dir_path}/feature_map.csv",
        # "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = Movielens100kRating()
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  num_examples = 0
  for x in train_dataset:
    num_examples += FLAGS.batch_size

  print(x)
  print(num_examples)


if __name__ == "__main__":
  app.run(runner)
