#!/usr/bin/env python
# @Time    : 2023/9/26 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys

from absl import app, flags

from deepray.datasets.criteo.criteo_tsv_reader import CriteoTsvReader
from deepray.utils.benchmark import PerformanceCalculator
import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de

de.enable_inference_mode()

FLAGS = flags.FLAGS


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=4096",
        "-epochs=1",
        f"--feature_map={dir_path}/feature_map_small.csv",
        "--model_dir=/results/tf_tfra_training_criteo_dcn_fp32_gbs4096_230926071440/export/",
        "--prefetch_buffer=64",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  loaded = tf.saved_model.load(FLAGS.model_dir)

  signature = loaded.signatures['serving_default']

  print(signature)

  _performance_calculator = PerformanceCalculator(0, 1000)
  num_examples = 0
  step = 0
  for x, y in train_dataset.take(1000):
    preds = loaded(x)

    step += 1
    num_examples += FLAGS.batch_size
    step_throughput = _performance_calculator(1, FLAGS.batch_size)

    if num_examples % 100 == 0:
      print(f'step {step}, Perf {step_throughput} samples/s')

  print(x)
  print(num_examples)
  results_perf = _performance_calculator.results
  if not _performance_calculator.completed:
    print(f"self._performance_calculator.completed: {_performance_calculator.completed}")
    results_perf = _performance_calculator.get_current_benchmark_results()
  print(results_perf)


if __name__ == "__main__":
  app.run(runner)