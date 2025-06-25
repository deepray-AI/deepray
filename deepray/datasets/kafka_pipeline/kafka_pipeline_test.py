# -*- coding: UTF-8 -*-
import sys
from absl import flags
import tensorflow as tf

from tf_keras import backend as K

import deepray as dp
from deepray.utils.benchmark import PerformanceCalculator
from deepray.utils import logging_util

from deepray.utils.horovod_utils import is_main_process

from deepray.datasets.kafka_pipeline.kafka_pipeline import KafkaPipeline

logger = logging_util.get_logger()


def main():

  data_pipe = KafkaPipeline(
      # dataset_name=flags.FLAGS.dataset,
      # partitions=[{'ds': date} for date in get_dates()],
  )

  train_dataset = data_pipe(input_file_pattern=None, batch_size=flags.FLAGS.batch_size)

  _performance_calculator = PerformanceCalculator(0, 1000)
  num_examples = 0
  step = 0

  for sample in train_dataset.take(1000):
    step += 1
    # example = tf.train.Example()
    # example.ParseFromString(sample[0].numpy())
    print(sample)
    # print(key)
    step_throughput = _performance_calculator(1, flags.FLAGS.batch_size)

    if num_examples % 100 == 0:
      logger.info(f'step {step}, Perf {step_throughput} samples/s')

  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  dp.runner(main)
