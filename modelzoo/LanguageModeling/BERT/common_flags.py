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

from absl import flags
import tensorflow as tf

from deepray.utils.flags import core as flags_core


def define_common_bert_flags():
  """Define common flags for BERT tasks."""
  flags.DEFINE_string("bert_config_file", None, "Bert configuration file to define core bert layers.")
  flags.DEFINE_string("model_export_path", None, "Path to the directory, where trainined model will be exported.")
  flags.DEFINE_string("tpu", "", "TPU address to connect to.")
  flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
  flags.DEFINE_integer(
    "steps_per_loop",
    200,
    "Number of steps per graph-mode loop. Only training step "
    "happens inside the loop. Callbacks will not be called "
    "inside.",
  )
  flags.DEFINE_boolean(
    "scale_loss", False, "Whether to divide the loss by number of replica inside the per-replica loss function."
  )
  flags.DEFINE_boolean(
    "use_keras_compile_fit",
    False,
    "If True, uses Keras compile/fit() API for training logic. Otherwise use custom training loop.",
  )
  flags.DEFINE_string(
    "hub_module_url", None, "TF-Hub path/url to Bert module. If specified, init_checkpoint flag should not be used."
  )
  flags.DEFINE_enum(
    "model_type",
    "bert",
    ["bert", "albert"],
    'Specifies the type of the model. If "bert", will use canonical BERT; if "albert", will use ALBERT model.',
  )
  flags.DEFINE_boolean("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
  flags.DEFINE_integer("save_checkpoint_steps", 1000, "save checkpoint for every n steps")
  flags.DEFINE_string("dllog_path", "bert_dllogger.json", "filename where dllogger writes to")
  flags.DEFINE_boolean("benchmark", False, "Benchmark mode.")


def use_float16():
  return flags_core.get_tf_dtype(flags.FLAGS) == tf.float16


def get_loss_scale():
  return flags_core.get_loss_scale(flags.FLAGS, default_for_fp16="dynamic")
