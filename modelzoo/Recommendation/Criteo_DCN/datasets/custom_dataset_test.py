#!/usr/bin/env python
# @Time    : 2023/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <fuhailin@outlook.com>
import sys

from absl import flags

from custom_dataset import CustomParquetPipeline
from deepray.utils.benchmark import PerformanceCalculator


def define_flags():
  argv = sys.argv + [
    "--batch_size=4096",
    "--epochs=1",
    # "--feature_map=feature_map_small.csv",
  ]
  flags.FLAGS(argv)


define_flags()
data_pipe = CustomParquetPipeline()
train_dataset = data_pipe(
  batch_size=flags.FLAGS.batch_size,
  input_file_pattern=[
    "/workspaces/datasets/00000-1-038360cf-9d9d-454c-8381-6a57bdbf6d57-00001.parquet",
    "/workspaces/datasets/01799-1800-26382079-2024-439e-84bf-e7b2231e0a2f-00001.parquet",
  ],
)
_performance_calculator = PerformanceCalculator(0, 1000)

num_examples = 0
step = 0
for x, y in train_dataset:
  step += 1
  num_examples += flags.FLAGS.batch_size
  step_throughput = _performance_calculator(1, flags.FLAGS.batch_size)

  if num_examples % 100 == 0:
    print(f"step {step}, Perf {step_throughput} samples/s")
# print(batch)

print(num_examples)
results_perf = _performance_calculator.results
if not _performance_calculator.completed:
  print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
  results_perf = _performance_calculator.get_current_benchmark_results()
print(results_perf)
