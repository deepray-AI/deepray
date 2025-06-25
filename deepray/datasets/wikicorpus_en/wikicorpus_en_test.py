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
from deepray.utils.benchmark import PerformanceCalculator

from .wikicorpus_en import Wikicorpus_en

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  if len(argv) <= 1:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    argv = [
      sys.argv[0],
      "--batch_size=128",
      "-epochs=1",
      f"--train_data=/workspaces/dataset/wikicorpus_en/data/tfrecord_lower_case_1_seq_len_128_random_seed_12345/wikicorpus_en/train/pretrain_data.*",
      # "--label=clicked",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = Wikicorpus_en(max_seq_length=128)
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)

  _performance_calculator = PerformanceCalculator(0, 1000)

  # partitions = data_pipe.get_supported_partitions()
  # print(partitions)
  num_examples = 0
  step = 0
  for batch in train_dataset:
    step += 1
    num_examples += FLAGS.batch_size
    step_throughput = _performance_calculator(1, FLAGS.batch_size)
    if num_examples % 100 == 0:
      print(f"num_examples {num_examples}, step {step}, Perf {step_throughput} samples/s")

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  app.run(runner)
