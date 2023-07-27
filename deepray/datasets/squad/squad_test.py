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

from .squad import Squad

FLAGS = flags.FLAGS

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

SQUAD_VERSION = "1.1"
# SQUAD_VERSION = "2.0"
SQUAD_DIR = f"/workspaces/bert_tf2/data/download/squad/v{SQUAD_VERSION}"


def runner(argv=None):
  if len(argv) <= 1:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    argv = [
        sys.argv[0],
        "--batch_size=1",
        "-epochs=1",
        f"--train_data={SQUAD_DIR}/squad_v{SQUAD_VERSION}_train.tf_record",
        f"--input_meta_data_path={dir_path}/v{SQUAD_VERSION}/squad_v{SQUAD_VERSION}_meta_data",
        # "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  data_pipe = Squad(max_seq_length=input_meta_data['max_seq_length'], dataset_type="squad")
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
