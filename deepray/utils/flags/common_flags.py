# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================
"""Defining common flags used across all BERT models/applications."""

import tensorflow as tf
from absl import flags

from deepray.utils.flags import core as flags_core


def define_common_flags():
  """Define common flags for BERT tasks."""
  flags_core.define_base(
    train_data=True,
    num_train_examples=True,
    batch_size=True,
    learning_rate=True,
    optimizer_type=True,
    use_custom_training_loop=True,
    num_accumulation_steps=True,
    init_checkpoint=True,
    num_gpus=True,
    model_dir=True,
    clean=True,
    epochs=True,
    stop_threshold=False,
    hooks=False,
    export_dir=False,
    run_eagerly=True,
  )
  flags.DEFINE_string(
    "config_file",
    default=None,
    help="YAML/JSON files which specifies overrides. The override order "
    "follows the order of args. Note that each file "
    "can be used as an override template to override the default parameters "
    "specified in Python. If the same parameter is specified in both "
    "`--config_file` and `--params_override`, `config_file` will be used "
    "first, followed by params_override.",
  )
  flags.DEFINE_integer(
    "steps_per_execution",
    None,
    "Number of steps per graph-mode loop. Only training step "
    "happens inside the loop. Callbacks will not be called "
    "inside.",
  )
  flags.DEFINE_integer("stop_steps", -1, "steps when training stops")
  flags.DEFINE_string(
    "model_name",
    None,
    'Specifies the name of the model. If "bert", will use canonical BERT; if "albert", will use ALBERT model.',
  )
  flags.DEFINE_bool("use_dynamic_embedding", False, "Whether use tfra.dynamic_embedding.")
  flags.DEFINE_integer(
    "random_seed", None, help=flags_core.help_wrap("This value will be used to seed both NumPy and TensorFlow.")
  )
  # Adds flags for mixed precision training.
  flags_core.define_performance(
    num_parallel_calls=False,
    inter_op=False,
    intra_op=False,
    synthetic_data=False,
    max_train_steps=False,
    dtype=True,
    dynamic_loss_scale=False,
    loss_scale=True,
    all_reduce_alg=False,
    num_packs=False,
    enable_xla=True,
    fp16_implementation=True,
  )

  flags_core.define_distribution(distribution_strategy=True)
  flags_core.define_data(
    dataset=True,
    data_dir=False,
    download_if_missing=False,
  )
  flags_core.define_device(tpu=False, redis=False)
  flags_core.define_benchmark()
  flags.DEFINE_float(
    "dropout_rate",
    default=-1,
    help="Dropout rate for all the classification MLPs (default: -1, disabled).",
  )
  flags.DEFINE_integer("prebatch", 1, "prebatch size for tfrecord")
  flags.DEFINE_string("feature_map", None, "path to feature_map")
  flags.DEFINE_string("black_list", None, "black list for feature_map")
  flags.DEFINE_string("white_list", None, "white list for feature_map")
  flags.DEFINE_integer("ev_slot_num", 0, "ev_slot_num")


def use_float16():
  return flags_core.get_tf_dtype(flags.FLAGS) == tf.float16


def get_loss_scale():
  return flags_core.get_loss_scale(flags.FLAGS, default_for_fp16="dynamic")
