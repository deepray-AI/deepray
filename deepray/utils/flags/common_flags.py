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
import datetime
import logging
import os

import tensorflow as tf
from absl import flags

from deepray.utils.flags import core as flags_core


def define_common_flags():
  """Define common flags for BERT tasks."""
  logging.info("flags base......................................")
  flags_core.define_base(
      train_data=True,
      num_train_examples=True,
      batch_size=True,
      learning_rate=True,
      optimizer_type=True,
      keras_use_ctl=True,
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
      dllog_path=True,
      save_checkpoint_steps=True,
  )
  flags.DEFINE_string(
      'config_file',
      default=None,
      help='YAML/JSON files which specifies overrides. The override order '
      'follows the order of args. Note that each file '
      'can be used as an override template to override the default parameters '
      'specified in Python. If the same parameter is specified in both '
      '`--config_file` and `--params_override`, `config_file` will be used '
      'first, followed by params_override.'
  )
  flags.DEFINE_string('vocab_file', None, 'The vocabulary file that the BERT model was trained on.')
  flags.DEFINE_bool(
      "do_lower_case", True, "Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models."
  )
  flags.DEFINE_string('model_export_path', None, 'Path to the directory, where trainined model will be '
                      'exported.')
  flags.DEFINE_integer(
      'steps_per_summary', 200, 'Number of steps per graph-mode loop. Only training step '
      'happens inside the loop. Callbacks will not be called '
      'inside.'
  )
  flags.DEFINE_integer("stop_steps", None, "steps when training stops")
  flags.DEFINE_boolean(
      'scale_loss', False, 'Whether to divide the loss by number of replica inside the per-replica '
      'loss function.'
  )
  flags.DEFINE_string(
      'hub_module_url', None, 'TF-Hub path/url to Bert module. '
      'If specified, init_checkpoint flag should not be used.'
  )
  flags.DEFINE_string(
      'model_type', None, 'Specifies the type of the model. '
      'If "bert", will use canonical BERT; if "albert", will use ALBERT model.'
  )
  flags.DEFINE_enum(
      'mode', 'train_and_predict',
      ['train_and_predict', 'train', 'predict', 'export_only', 'sm_predict', 'trt_predict'],
      'One of {"train_and_predict", "train", "predict", "export_only", "sm_predict", "trt_predict"}. '
      '`train_and_predict`: both train and predict to a json file. '
      '`train`: only trains the model. '
      'trains the model and evaluates in the meantime. '
      '`predict`: predict answers from the squad json file. '
      '`export_only`: will take the latest checkpoint inside '
      'model_dir and export a `SavedModel`.'
      '`sm_predict`: will load SavedModel from savedmodel_dir and predict answers'
      '`trt_predict`: will load SavedModel from savedmodel_dir, convert and predict answers with TF-TRT'
  )
  flags.DEFINE_string(
      'input_meta_data_path', None, 'Path to file that contains meta data about input '
      'to be used for training and evaluation.'
  )
  flags.DEFINE_bool("use_dynamic_embedding", False, "Whether use tfra.dynamic_embedding.")
  flags.DEFINE_string('predict_file', None, 'Prediction data path with train tfrecords.')
  flags.DEFINE_string(
      "eval_script", None, "SQuAD evaluate.py file to compute f1 and exact_match E.g., evaluate-v1.1.py"
  )
  flags.DEFINE_integer(
      'n_best_size', 20, 'The total number of n-best predictions to generate in the '
      'nbest_predictions.json output file.'
  )
  flags.DEFINE_integer(
      'max_answer_length', 30, 'The maximum length of an answer that can be generated. This is needed '
      'because the start and end predictions are not conditioned on one another.'
  )
  flags.DEFINE_bool(
      'verbose_logging', False, 'If true, all of the warnings related to data processing will be printed. '
      'A number of warnings are expected for a normal SQuAD evaluation.'
  )
  flags.DEFINE_integer(
      "random_seed", 12345, help=flags_core.help_wrap("This value will be used to seed both NumPy and TensorFlow.")
  )
  # Adds flags for mixed precision training.
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=True,
      dynamic_loss_scale=True,
      loss_scale=True,
      all_reduce_alg=False,
      num_packs=False,
      enable_xla=True,
      fp16_implementation=True,
  )

  flags_core.define_distribution(distribution_strategy=True)
  flags_core.define_data(
      dataset=True,
      data_dir=True,
      download_if_missing=True,
  )
  flags_core.define_device(tpu=False, redis=True)
  flags_core.define_benchmark(benchmark=True,)

  flags.DEFINE_string(
      name="date", default=(datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"), help=""
  )
  flags.DEFINE_string(name="restore_date", default=None, help="")
  flags.DEFINE_string(name="start_date", default=None, help="")
  flags.DEFINE_string(name="end_date", default=None, help="")
  flags.DEFINE_string(name="fine_tune", default=None, help="")
  flags.DEFINE_string(name="warmup_path", default=None, help="")
  flags.DEFINE_float(
      "dropout_rate",
      default=-1,
      help="Dropout rate for all the classification MLPs (default: -1, disabled).",
  )
  flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
  flags.DEFINE_integer("prebatch", 1, "prebatch size for tfrecord")
  flags.DEFINE_list("label", [], "label name")
  flags.DEFINE_string("feature_map", os.path.join(os.getcwd(), "business/data/feature_map.csv"), "path to feature_map")
  flags.DEFINE_string("black_list", None, "black list for feature_map")
  flags.DEFINE_string("white_list", None, "white list for feature_map")


def use_float16():
  return flags_core.get_tf_dtype(flags.FLAGS) == tf.float16


def get_loss_scale():
  return flags_core.get_loss_scale(flags.FLAGS, default_for_fp16='dynamic')
