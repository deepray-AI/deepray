#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys

from absl import flags

import deepray as dp
from deepray.datasets.parquet_pipeline.ali_parquet_dataset import ParquetPipeline
from deepray.utils.benchmark import PerformanceCalculator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def define_flags():
  argv = sys.argv + [
    "--batch_size=4096",
    "--epochs=1",
    "--dataset=ps_test",
    "--feature_map=/workspaces/one-code/shadow-tf/datasets/feature_map.csv",
    "--config_file=/workspaces/one-code/shadow-tf/train_feature_process.yaml",
  ]
  flags.FLAGS(argv)


def main():
  define_flags()
  filenames = [
    "/workspaces/datasets/00000-1-038360cf-9d9d-454c-8381-6a57bdbf6d57-00001.parquet",
    "/workspaces/datasets/01799-1800-26382079-2024-439e-84bf-e7b2231e0a2f-00001.parquet",
  ]
  data_pipe = ParquetPipeline(column_names=["f_c0", "f_c1", "f_c14"])
  # create data pipline of train & test dataset
  train_dataset = data_pipe(batch_size=flags.FLAGS.batch_size, input_file_pattern=filenames, is_training=True)
  _performance_calculator = PerformanceCalculator(0, 1000)

  num_examples = 0
  step = 0
  for batch in train_dataset.take(1000):
    step += 1
    num_examples += flags.FLAGS.batch_size
    step_throughput = _performance_calculator(1, flags.FLAGS.batch_size)
    print(f"step {step}, Perf {step_throughput} samples/s")
  print(batch)

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  dp.runner(main)
