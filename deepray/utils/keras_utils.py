# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for the Keras implementations of models."""

import multiprocessing
import os

import tensorflow as tf
from absl import logging
from tensorflow.python import tf2

from deepray.utils import logging_util

logger = logging_util.get_logger()


def set_session_config(enable_eager=False, enable_xla=False):
  """Sets the session config."""
  if is_v2_0():
    set_config_v2(enable_xla=enable_xla)
  else:
    config = get_config_proto_v1(enable_xla=enable_xla)
    if enable_eager:
      tf.compat.v1.enable_eager_execution(config=config)
    else:
      sess = tf.Session(config=config)
      tf.keras.backend.set_session(sess)


def get_config_proto_v1(enable_xla=False):
  """Return config proto according to flag settings, or None to use default."""
  config = None
  if enable_xla:
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
  return config


def set_config_v2(enable_xla=False):
  """Config eager context according to flag values using TF 2.0 API."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)
    logger.info("XLA activated")


def is_v2_0():
  """Returns true if using tf 2.0."""
  return tf2.enabled()


def set_gpu_thread_mode_and_count(gpu_thread_mode, datasets_num_private_threads, num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info("Logical CPU cores: %s", cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ["TF_GPU_THREAD_MODE"] = gpu_thread_mode
  os.environ["TF_GPU_THREAD_COUNT"] = str(per_gpu_thread_count)
  logging.info("TF_GPU_THREAD_COUNT: %s", os.environ["TF_GPU_THREAD_COUNT"])
  logging.info("TF_GPU_THREAD_MODE: %s", os.environ["TF_GPU_THREAD_MODE"])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(cpu_count - total_gpu_thread_count - num_runtime_threads, num_gpus * 8)
    logging.info("Set datasets_num_private_threads to %s", datasets_num_private_threads)
