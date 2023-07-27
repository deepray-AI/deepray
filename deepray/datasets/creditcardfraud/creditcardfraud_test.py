#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import sys
from datetime import datetime

from absl import app, flags

from .creditcardfraud import CreditCardFraud

FLAGS = flags.FLAGS

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=10",
        "-epochs=1",
        "--train_data=movielens/1m-ratings",
        # f"--feature_map={dir_path}/feature_map.csv",
        "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = CreditCardFraud()
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  num_examples = 0
  for x, y in train_dataset:
    num_examples += FLAGS.batch_size

  print(x)
  print(num_examples)


if __name__ == "__main__":
  app.run(runner)
