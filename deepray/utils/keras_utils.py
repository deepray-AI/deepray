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
from deepray.utils.horovod_utils import get_world_size, main_print, allreduce

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


def format_param_count(count):
  """Format parameter count into human-readable string."""
  count = int(count)
  if count < 1000:
    return f"{count}"
  elif count < 1_000_000:
    return f"{count / 1_000:,.1f}K"
  elif count < 1_000_000_000:
    return f"{count / 1_000_000:,.1f}M"
  elif count < 1_000_000_000_000:
    return f"{count / 1_000_000_000:,.2f}B"
  else:
    return f"{count / 1_000_000_000_000:,.2f}T"


def count_params(model):
  """Count the total number of parameters in a model, including EmbeddingVariables."""
  from tf_keras.src.utils.layer_utils import count_params
  from deepray.custom_ops.embedding_variable import kv_variable_ops

  model_size = 0
  regular_weights = []
  embedding_vars = []

  for weight in model.weights:
    if isinstance(weight, kv_variable_ops.EmbeddingVariable):
      # Calculate parameters for EmbeddingVariable
      shape = weight.get_dynamic_shape()
      param_count = shape[0] * shape[1]
      if get_world_size() > 1:
        param_count = allreduce(param_count, op="sum")
      model_size += param_count
      embedding_vars.append((weight.name, int(shape[0]), int(shape[1]), param_count))
    else:
      # Collect regular weights
      regular_weights.append(weight)

  # Print embedding variables sorted by first dimension
  if embedding_vars:
    main_print("\nEmbedding Variables (sorted by vocabulary size):")
    main_print("-" * 78)
    main_print(f"{'Name':30} | {'Shape(IDs * Dims)':18} | {'Params':>10} | {'% Total':>7}")
    main_print("-" * 78)

    embedding_vars.sort(key=lambda x: x[1], reverse=True)
    total_embedding_params = sum(param_count for _, _, _, param_count in embedding_vars)

    for name, num_ids, num_dims, param_count in embedding_vars:
      percentage = (param_count / total_embedding_params) * 100
      shape_str = f"{num_ids:,} * {num_dims}"
      # Shorten the long names
      short_name = name if len(name) <= 38 else name[:35] + "..."
      main_print(f"{short_name:30} | {shape_str:18} | {format_param_count(param_count):>10} | {percentage:5.1f}%")

  # Add parameters from regular weights
  regular_params = count_params(regular_weights)
  model_size += regular_params

  # Print summary with better formatting
  main_print("\nParameter Summary:")
  main_print("-" * 50)
  embedding_total = model_size - regular_params
  embedding_percent = (embedding_total / model_size * 100) if model_size > 0 else 0
  regular_percent = (regular_params / model_size * 100) if model_size > 0 else 0

  main_print(f"{'Embedding parameters:':20} {format_param_count(embedding_total):>10} ({embedding_percent:5.1f}%)")
  main_print(f"{'Regular parameters:':20} {format_param_count(regular_params):>10} ({regular_percent:5.1f}%)")
  main_print("-" * 50)
  main_print(f"{'TOTAL:':20} {format_param_count(model_size):>10}")
  main_print(f"{'':20} ({model_size:,})")

  return model_size
