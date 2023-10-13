#!/usr/bin/env python
# @Time    : 2023/9/26 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys
import tensorflow as tf
import numpy as np
from absl import app, flags

from deepray.datasets.criteo.criteo_tsv_reader import CriteoTsvReader
from deepray.utils.benchmark import PerformanceCalculator
from deepray.utils.export import SavedModel

FLAGS = flags.FLAGS


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=4096",
        "--run_eagerly=True",
        "--use_dynamic_embedding=True",
        f"--feature_map={dir_path}/feature_map_small.csv",
        "--model_dir=/workspaces/tmp/export_main_optimized/",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  # create data pipline of train & test dataset
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  model = SavedModel(FLAGS.model_dir, "amp" if FLAGS.dtype else "fp32")
  signature = model.saved_model_loaded.signatures['serving_default']
  print(signature)

  _performance_calculator = PerformanceCalculator(0, 1000)
  num_examples = 0
  step = 0

  a = {
      "feature_14":
          tf.constant(np.array([6394203, 7535249, 3500077, 836339, 7401745, 375123]), dtype=tf.int32),
      "feature_15":
          tf.constant(np.array([6394203, 7535249, 3500077, 836339, 7401745, 375123]), dtype=tf.int32),
      "dense_features":
          tf.constant(
              np.array(
                  [
                      [0.7361634, 0.7361634], [0.00337589, 0.00337589], [0.673707, 0.673707], [0.33169293, 0.33169293],
                      [0.8020003, 0.8020003], [0.18556607, 0.18556607]
                  ]
              ),
              dtype=tf.float32
          )
  }

  b = {
      "feature_14": tf.constant(np.array([1]), dtype=tf.int32),
      "feature_15": tf.constant(np.array([1]), dtype=tf.int32),
      "dense_features": tf.constant(np.array([[1.0, 1.0]]), dtype=tf.float32)
  }

  print(model(a))
  print(model(b))
  exit(0)

  for x, y in train_dataset.take(300):
    preds = model(x)
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
