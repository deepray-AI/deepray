#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys
from datetime import datetime

from absl import app, flags

from deepray.utils.benchmark import PerformanceCalculator
from .amazon_books_2014 import AmazonBooks2014

FLAGS = flags.FLAGS

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=128",
        "-epochs=1",
        "--train_data=/workspaces/dataset/amazon_books_2014/tfrecord_path/train/*.tfrecord",
        "--max_seq_length=90",
        f"--feature_map={dir_path}/feature_map.csv",
        # "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = AmazonBooks2014(FLAGS.max_seq_length)
  # create data pipline of train & test dataset

  # since each tfrecord file must include all of the features, it is enough to read first chunk for each split.
  # train_files = [dataset_dir / file for file in feature_spec.source_spec[TRAIN_MAPPING][0][FILES_SELECTOR]]

  prebatch_size = 5
  train_dataset = data_pipe(FLAGS.train_data, batch_size=FLAGS.batch_size, prebatch_size=prebatch_size)

  _performance_calculator = PerformanceCalculator(0, 1000)

  # partitions = data_pipe.get_supported_partitions()
  # print(partitions)
  num_examples = 0
  step = 0
  for batch in train_dataset:
    step += 1
    num_examples += FLAGS.batch_size
    step_throughput = _performance_calculator(1, FLAGS.batch_size)
    print(f'step {step}, Perf {step_throughput} samples/s')

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  app.run(runner)
