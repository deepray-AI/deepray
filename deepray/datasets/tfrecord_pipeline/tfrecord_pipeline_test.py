#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import sys
import os
from datetime import datetime

from absl import app, flags

from deepray.datasets.tfrecord_pipeline import TFRecordPipeline
from deepray.utils.benchmark import PerformanceCalculator

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))

  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=256",
        "--epochs=1",
        "--prebatch=1",
        "--train_data=/workspaces/dataset/*.tfrecord",
        "--feature_map=feature_map.csv",
        "--label=label",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = TFRecordPipeline()
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
    print(f'step {step}, Perf {step_throughput} samples/s')
  # print(batch)

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  app.run(runner)
