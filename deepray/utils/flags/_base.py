# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Flags which will be nearly universal across models."""

import datetime
import sys

from absl import flags

from deepray.utils.flags import core as flags_core
from deepray.utils.flags._conventions import help_wrap


def define_base(
  train_data=False,
  num_train_examples=False,
  learning_rate=False,
  optimizer_type=False,
  use_custom_training_loop=False,
  model_dir=False,
  clean=False,
  num_accumulation_steps=False,
  epochs=False,
  stop_threshold=False,
  batch_size=False,
  num_gpus=False,
  init_checkpoint=False,
  hooks=False,
  export_dir=False,
  run_eagerly=False,
):
  """Register base flags.

  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    clean: Create a flag for removing the model_dir.
    epochs: Create a flag to specify the number of training epochs.
    stop_threshold: Create a flag to specify a threshold accuracy or other eval
      metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    num_gpus: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.
    run_eagerly: Create a flag to specify to run eagerly op by op.

  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []
  if train_data:
    flags.DEFINE_list("train_data", None, "File paths or regular expression to match train files.")
    flags.DEFINE_list("valid_data", None, "File paths or regular expression to match validation files.")
    flags.DEFINE_list("test_data", None, "File paths or regular expression to match test files.")
    key_flags.append("train_data")
    key_flags.append("valid_data")
    key_flags.append("test_data")
  if num_train_examples:
    flags.DEFINE_integer("num_train_examples", -1, "number of training examples.")
    flags.DEFINE_integer("num_valid_examples", -1, "number of validation examples.")
    key_flags.append("num_train_examples")
  if learning_rate:
    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
    key_flags.append("learning_rate")
  if use_custom_training_loop:
    flags.DEFINE_bool(
      name="use_custom_training_loop",
      default=True,
      help=flags_core.help_wrap("If True, we use a custom training loop for keras."),
    )
    key_flags.append("use_custom_training_loop")
  if optimizer_type:
    flags.DEFINE_string("optimizer_type", "adam", "Optimizer used for training - LAMB or ADAM")
    key_flags.append("optimizer_type")
  if num_accumulation_steps:
    flags.DEFINE_integer("num_accumulation_steps", 1, "Number of accumulation steps before gradient update.")
    key_flags.append("num_accumulation_steps")

  if init_checkpoint:
    flags.DEFINE_list("init_checkpoint", "", "Initial checkpoint (usually from a pre-trained BERT model).")
    key_flags.append("init_checkpoint")
    flags.DEFINE_list("init_weights", "", "Initial weights for the main model.")
    key_flags.append("init_weights")

  if model_dir:
    flags.DEFINE_string(
      name="model_dir",
      default=f"/tmp/{datetime.datetime.now().timestamp()}",
      help=help_wrap("The location of the model checkpoint files."),
    )
    key_flags.append("model_dir")

  if clean:
    flags.DEFINE_boolean(name="clean", default=False, help=help_wrap("If set, model_dir will be removed if it exists."))
    key_flags.append("clean")

  if epochs:
    flags.DEFINE_integer(
      name="steps_per_epoch", default=None, help=help_wrap("The number of steps in one epoch used to train.")
    )
    flags.DEFINE_integer(name="epochs", default=1, help=help_wrap("The number of epochs used to train."))
    key_flags.append("steps_per_epoch")
    key_flags.append("epochs")

  if stop_threshold:
    flags.DEFINE_float(
      name="stop_threshold",
      default=None,
      help=help_wrap(
        "If passed, training will stop at the earlier of "
        "epochs and when the evaluation metric is  "
        "greater than or equal to stop_threshold."
      ),
    )

  if batch_size:
    flags.DEFINE_integer(
      name="batch_size",
      default=32,
      help=help_wrap(
        "Batch size for training and evaluation. When using "
        "multiple gpus, this is the global batch size for "
        "all devices. For example, if the batch size is 32 "
        "and there are 4 GPUs, each GPU will get 8 examples on "
        "each step."
      ),
    )
    flags.DEFINE_integer(
      name="eval_batch_size",
      default=None,
      help=help_wrap(
        "The batch size used for evaluation. This should generally be larger"
        "than the training batch size as the lack of back propagation during"
        "evaluation can allow for larger batch sizes to fit in memory. If not"
        "specified, the training batch size (--batch_size) will be used."
      ),
    )
    key_flags.append("batch_size")

  if num_gpus:
    flags.DEFINE_integer(
      name="num_gpus",
      default=0,
      help=help_wrap("How many GPUs to use at each worker with the DistributionStrategies API. The default is 1."),
    )

  if run_eagerly:
    flags.DEFINE_boolean(
      name="run_eagerly", default=False, help="Run the model op by op without building a model function."
    )

  if hooks:
    flags.DEFINE_list(
      name="hooks",
      default="LoggingTensorHook",
      help=help_wrap(
        "A list of (case insensitive) strings to specify the names of "
        "training hooks. Example: `--hooks ProfilerHook,"
        "ExamplesPerSecondHook`\n See hooks_helper "
        "for details."
      ),
    )
    key_flags.append("hooks")

  if export_dir:
    flags.DEFINE_string(
      name="export_dir",
      default=None,
      help=help_wrap(
        "If set, a SavedModel serialization of the model will "
        "be exported to this directory at the end of training. "
        "See the README for more details and relevant links."
      ),
    )
    key_flags.append("export_dir")

  return key_flags


def get_num_gpus(flags_obj):
  """Treat num_gpus=-1 as 'use all'."""
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])
