#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys
from datetime import datetime

from absl import app, flags

from deepray.utils.benchmark import PerformanceCalculator
from .parquet_pipeline import parquet_pipeline

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
      sys.argv[0],
      "--batch_size=2",
      "--epochs=1",
      "--train_data=/workspaces/dataset/arc13_training_v3_data/*.parquet",
      "--feature_map=examples/Recommendation/yekuan/tools/feature_map.csv",
      # "--white_list=examples/Recommendation/yekuan/data_pipeline/white_list",
      # f"--feature_map={dir_path}/bz_search_1to3.csv",
      "--label=deal_type,detailed_deal_type",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = parquet_pipeline()
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  _performance_calculator = PerformanceCalculator(0, 1000)

  # partitions = data_pipe.get_supported_partitions()
  # print(partitions)
  num_examples = 0
  step = 0
  for batch in train_dataset.take(1000):
    step += 1
    num_examples += FLAGS.batch_size
    step_throughput = _performance_calculator(1, FLAGS.batch_size)
    print(f"step {step}, Perf {step_throughput} samples/s")

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  app.run(runner)
