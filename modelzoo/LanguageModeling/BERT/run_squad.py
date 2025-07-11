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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import time

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from absl import flags
from absl import logging

import common_flags
import deepray as dp
import input_pipeline
import model_saving_utils
import optimization

# word-piece tokenizer based squad_lib
import squad_lib as squad_lib_wp

# sentence-piece tokenizer based squad_lib
import squad_lib_sp
import tf_trt
import tokenization
from deepray.core.common import distribution_utils
from deepray.core.trainer import Trainer
from deepray.datasets.squad import Squad
from deepray.utils import keras_utils
from official.nlp import bert_modeling as modeling

# Import BERT model libraries.
from official.nlp import bert_models


MODEL_CLASSES = {
  "bert": (modeling.BertConfig, squad_lib_wp, tokenization.FullTokenizer),
  "albert": (modeling.AlbertConfig, squad_lib_sp, tokenization.FullSentencePieceTokenizer),
}


def define_flags():
  flags.DEFINE_enum(
    "mode",
    "train_and_predict",
    ["train_and_predict", "train", "predict", "export_only", "sm_predict", "trt_predict"],
    'One of {"train_and_predict", "train", "predict", "export_only", "sm_predict", "trt_predict"}. '
    "`train_and_predict`: both train and predict to a json file. "
    "`train`: only trains the model. "
    "trains the model and evaluates in the meantime. "
    "`predict`: predict answers from the squad json file. "
    "`export_only`: will take the latest checkpoint inside "
    "model_dir and export a `SavedModel`."
    "`sm_predict`: will load SavedModel from savedmodel_dir and predict answers"
    "`trt_predict`: will load SavedModel from savedmodel_dir, convert and predict answers with TF-TRT",
  )
  flags.DEFINE_string("train_data_path", "", "Training data path with train tfrecords.")
  flags.DEFINE_string(
    "input_meta_data_path",
    None,
    "Path to file that contains meta data about input to be used for training and evaluation.",
  )
  flags.DEFINE_string(
    "eval_script", None, "SQuAD evaluate.py file to compute f1 and exact_match E.g., evaluate-v1.1.py"
  )

  # Model training specific flags.
  flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")
  # Predict processing related.
  flags.DEFINE_string("predict_file", None, "Prediction data path with train tfrecords.")
  flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
  flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased models and False for cased models.",
  )
  flags.DEFINE_bool(
    "verbose_logging",
    False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
  )
  flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for prediction.")
  flags.DEFINE_integer(
    "n_best_size", 20, "The total number of n-best predictions to generate in the nbest_predictions.json output file."
  )
  flags.DEFINE_integer(
    "max_answer_length",
    30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.",
  )
  flags.DEFINE_string(
    "sp_model_file", None, "The path to the sentence piece model. Used by sentence piece tokenizer employed by ALBERT."
  )
  flags.DEFINE_string("savedmodel_dir", None, "The path of SavedModel for Savedmodel and TF-TRT prediction.")

  common_flags.define_common_bert_flags()
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("model_dir")
  flags.FLAGS(sys.argv)


def squad_loss_fn(start_positions, end_positions, start_logits, end_logits, loss_factor=1.0):
  """Returns sparse categorical crossentropy for start/end logits."""
  start_loss = tf.keras.backend.sparse_categorical_crossentropy(start_positions, start_logits, from_logits=True)
  end_loss = tf.keras.backend.sparse_categorical_crossentropy(end_positions, end_logits, from_logits=True)

  total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
  total_loss *= loss_factor
  return total_loss


def get_loss_fn(loss_factor=1.0):
  """Gets a loss function for squad task."""

  def _loss_fn(labels, model_outputs):
    start_positions = labels["start_positions"]
    end_positions = labels["end_positions"]
    start_logits, end_logits = model_outputs
    return squad_loss_fn(start_positions, end_positions, start_logits, end_logits, loss_factor=loss_factor)

  return _loss_fn


def get_raw_results(predictions):
  """Converts multi-replica predictions to RawResult."""
  squad_lib = MODEL_CLASSES[flags.FLAGS.model_type][1]
  for unique_ids, start_logits, end_logits in zip(
    predictions["unique_ids"], predictions["start_logits"], predictions["end_logits"]
  ):
    for values in zip(unique_ids.numpy(), start_logits.numpy(), end_logits.numpy()):
      yield squad_lib.RawResult(unique_id=values[0], start_logits=values[1].tolist(), end_logits=values[2].tolist())


def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training, use_horovod):
  """Gets a closure to create a dataset.."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_squad_dataset(
      input_file_pattern,
      max_seq_length,
      batch_size,
      is_training=is_training,
      input_pipeline_context=ctx,
      use_horovod=use_horovod,
    )
    return dataset

  return _dataset_fn


def predict_squad_customized(strategy, input_meta_data, bert_config, predict_tfrecord_path, num_steps):
  """Make predictions using a Bert-based squad model."""
  predict_dataset_fn = get_dataset_fn(
    predict_tfrecord_path,
    input_meta_data["max_seq_length"],
    flags.FLAGS.predict_batch_size,
    is_training=False,
    use_horovod=False,
  )
  if strategy:
    predict_iterator = iter(strategy.experimental_distribute_datasets_from_function(predict_dataset_fn))
  else:
    predict_iterator = iter(predict_dataset_fn())

  if flags.FLAGS.mode == "trt_predict":
    squad_model = tf_trt.TFTRTModel(flags.FLAGS.savedmodel_dir, "amp" if flags.FLAGS.use_fp16 else "fp32")

  elif flags.FLAGS.mode == "sm_predict":
    squad_model = tf_trt.SavedModel(flags.FLAGS.savedmodel_dir, "amp" if flags.FLAGS.use_fp16 else "fp32")

  else:
    with distribution_utils.get_strategy_scope(strategy):
      squad_model, _ = bert_models.squad_model(
        bert_config, input_meta_data["max_seq_length"], float_type=tf.float16 if flags.FLAGS.use_fp16 else tf.float32
      )

    if flags.FLAGS.init_checkpoint:
      checkpoint = tf.train.Checkpoint(model=squad_model)
      checkpoint.restore(flags.FLAGS.init_checkpoint[0]).expect_partial()

    checkpoint_path = tf.train.latest_checkpoint(flags.FLAGS.model_dir)
    logging.info("Restoring checkpoints from %s", checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path).expect_partial()

  @tf.function
  def predict_step(iterator):
    """Predicts on distributed devices."""

    def _replicated_step(inputs):
      """Replicated prediction calculation."""
      x, _ = inputs
      unique_ids = x.pop("unique_ids")
      if flags.FLAGS.benchmark:
        t0 = tf.timestamp()
        unique_ids = t0
      start_logits, end_logits = squad_model(x, training=False)
      return dict(unique_ids=unique_ids, start_logits=start_logits, end_logits=end_logits)

    def tuple_fun(x):
      return (x,)

    if strategy:
      outputs = strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))
      map_func = strategy.experimental_local_results
    else:
      outputs = _replicated_step(
        next(iterator),
      )
      map_func = tuple_fun
    return tf.nest.map_structure(map_func, outputs)

  all_results = []
  time_list = []
  eval_start_time = time.time()
  elapsed_secs = 0

  for _ in range(num_steps):
    predictions = predict_step(predict_iterator)
    if flags.FLAGS.benchmark:
      # transfer tensor to CPU for synchronization
      t0 = predictions["unique_ids"][0]
      start_logits = predictions["start_logits"][0]
      start_logits.numpy()
      elapsed_secs = time.time() - t0.numpy()
      # Removing first 4 (arbitrary) number of startup iterations from perf evaluations
      if _ > 3:
        time_list.append(elapsed_secs)
      continue

    for result in get_raw_results(predictions):
      all_results.append(result)

    if len(all_results) % 100 == 0:
      logging.info("Made predictions for %d records.", len(all_results))

  eval_time_elapsed = time.time() - eval_start_time
  logging.info("-----------------------------")
  logging.info("Summary Inference Statistics")
  logging.info("Batch size = %d", flags.FLAGS.predict_batch_size)
  logging.info("Sequence Length = %d", input_meta_data["max_seq_length"])
  logging.info("Precision = %s", "fp16" if flags.FLAGS.use_fp16 else "fp32")
  logging.info(
    "Total Inference Time = %0.2f for Sentences = %d", eval_time_elapsed, num_steps * flags.FLAGS.predict_batch_size
  )

  if flags.FLAGS.benchmark:
    eval_time_wo_overhead = sum(time_list)
    time_list.sort()
    num_sentences = (num_steps - 4) * flags.FLAGS.predict_batch_size

    avg = np.mean(time_list)
    cf_50 = max(time_list[: int(len(time_list) * 0.50)])
    cf_90 = max(time_list[: int(len(time_list) * 0.90)])
    cf_95 = max(time_list[: int(len(time_list) * 0.95)])
    cf_99 = max(time_list[: int(len(time_list) * 0.99)])
    cf_100 = max(time_list[: int(len(time_list) * 1)])
    ss_sentences_per_second = num_sentences * 1.0 / eval_time_wo_overhead

    logging.info(
      "Total Inference Time W/O Overhead = %0.2f for Sequences = %d",
      eval_time_wo_overhead,
      (num_steps - 4) * flags.FLAGS.predict_batch_size,
    )
    logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
    logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
    logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
    logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
    logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
    logging.info("Latency Average (ms) = %0.2f", avg * 1000)
    logging.info("Throughput Average (sequences/sec) = %0.2f", ss_sentences_per_second)

  logging.info("-----------------------------")

  return all_results


def train_squad(strategy, input_meta_data, custom_callbacks=None):
  """Run bert squad training."""
  if strategy:
    logging.info("Training using customized training loop with distribution strategy.")
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_config_v2(flags.FLAGS.enable_xla)

  use_float16 = common_flags.use_float16()
  if use_float16:
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")

  bert_config = MODEL_CLASSES[flags.FLAGS.model_type][0].from_json_file(flags.FLAGS.bert_config_file)
  num_train_examples = input_meta_data["train_data_size"]
  max_seq_length = input_meta_data["max_seq_length"]
  global_batch_size = flags.FLAGS.train_batch_size * flags.FLAGS.num_accumulation_steps
  if flags.FLAGS.use_horovod:
    global_batch_size *= hvd.size()
  steps_per_epoch = int(num_train_examples / global_batch_size)
  warmup_steps = int(flags.FLAGS.epochs * num_train_examples * 0.1 / global_batch_size)

  data_pipe = Squad(max_seq_length=max_seq_length, dataset_type="squad")
  train_input = data_pipe(
    flags.FLAGS.batch_size,
    flags.FLAGS.train_data_path,
    is_training=True,
  )

  squad_model, core_model = bert_models.squad_model(
    bert_config,
    max_seq_length,
    float_type=tf.float16 if flags.FLAGS.use_fp16 else tf.float32,
    hub_module_url=flags.FLAGS.hub_module_url,
  )

  learning_rate = flags.FLAGS.learning_rate * hvd.size() if flags.FLAGS.use_horovod else flags.FLAGS.learning_rate
  optimizer = optimization.create_optimizer(
    learning_rate, steps_per_epoch * flags.FLAGS.epochs, warmup_steps, flags.FLAGS.optimizer_type
  )
  if flags.FLAGS.use_fp16:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

  trainer = Trainer(
    model={"main": squad_model, "sub_model": core_model},
    optimizer=optimizer,
    loss={
      "start_positions": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      "end_positions": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    },
    loss_weights={"start_positions": 0.5, "end_positions": 0.5},
    jit_compile=True,
  )
  trainer.fit(x=train_input, steps_per_epoch=steps_per_epoch)


def predict_squad(strategy, input_meta_data):
  """Makes predictions for a squad dataset."""
  keras_utils.set_config_v2(flags.FLAGS.enable_xla)
  config_cls, squad_lib, tokenizer_cls = MODEL_CLASSES[flags.FLAGS.model_type]
  bert_config = config_cls.from_json_file(flags.FLAGS.bert_config_file)
  if tokenizer_cls == tokenization.FullTokenizer:
    tokenizer = tokenizer_cls(vocab_file=flags.FLAGS.vocab_file, do_lower_case=flags.FLAGS.do_lower_case)
  else:
    assert tokenizer_cls == tokenization.FullSentencePieceTokenizer
    tokenizer = tokenizer_cls(sp_model_file=flags.FLAGS.sp_model_file)
  doc_stride = input_meta_data["doc_stride"]
  max_query_length = input_meta_data["max_query_length"]
  # Whether data should be in Ver 2.0 format.
  version_2_with_negative = input_meta_data.get("version_2_with_negative", False)
  eval_examples = squad_lib.read_squad_examples(
    input_file=flags.FLAGS.predict_file, is_training=False, version_2_with_negative=version_2_with_negative
  )

  eval_writer = squad_lib.FeatureWriter(
    filename=os.path.join(flags.FLAGS.model_dir, "eval.tf_record"), is_training=False
  )
  eval_features = []

  def _append_feature(feature, is_padding):
    if not is_padding:
      eval_features.append(feature)
    eval_writer.process_feature(feature)

  # TPU requires a fixed batch size for all batches, therefore the number
  # of examples must be a multiple of the batch size, or else examples
  # will get dropped. So we pad with fake examples which are ignored
  # later on.
  kwargs = dict(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=input_meta_data["max_seq_length"],
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    output_fn=_append_feature,
    batch_size=flags.FLAGS.predict_batch_size,
  )

  # squad_lib_sp requires one more argument 'do_lower_case'.
  if squad_lib == squad_lib_sp:
    kwargs["do_lower_case"] = flags.FLAGS.do_lower_case
  dataset_size = squad_lib.convert_examples_to_features(**kwargs)
  eval_writer.close()

  logging.info("***** Running predictions *****")
  logging.info("  Num orig examples = %d", len(eval_examples))
  logging.info("  Num split examples = %d", len(eval_features))
  logging.info("  Batch size = %d", flags.FLAGS.predict_batch_size)

  num_steps = int(dataset_size / flags.FLAGS.predict_batch_size)
  if flags.FLAGS.benchmark and num_steps > 1000:
    num_steps = 1000
  all_results = predict_squad_customized(strategy, input_meta_data, bert_config, eval_writer.filename, num_steps)

  if flags.FLAGS.benchmark:
    return

  output_prediction_file = os.path.join(flags.FLAGS.model_dir, "predictions.json")
  output_nbest_file = os.path.join(flags.FLAGS.model_dir, "nbest_predictions.json")
  output_null_log_odds_file = os.path.join(flags.FLAGS.model_dir, "null_odds.json")

  squad_lib.write_predictions(
    eval_examples,
    eval_features,
    all_results,
    flags.FLAGS.n_best_size,
    flags.FLAGS.max_answer_length,
    flags.FLAGS.do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose=flags.FLAGS.verbose_logging,
  )

  if flags.FLAGS.eval_script:
    eval_out = subprocess.check_output([
      sys.executable,
      flags.FLAGS.eval_script,
      flags.FLAGS.predict_file,
      output_prediction_file,
    ])
    scores = str(eval_out).strip()
    exact_match = float(scores.split(":")[1].split(",")[0])
    if version_2_with_negative:
      f1 = float(scores.split(":")[2].split(",")[0])
    else:
      f1 = float(scores.split(":")[2].split("}")[0])
    print(str(eval_out))


def export_squad(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError("Export path is not specified: %s" % model_export_path)
  bert_config = MODEL_CLASSES[flags.FLAGS.model_type][0].from_json_file(flags.FLAGS.bert_config_file)
  squad_model, _ = bert_models.squad_model(bert_config, input_meta_data["max_seq_length"], float_type=tf.float32)
  model_saving_utils.export_bert_model(
    model_export_path + "/savedmodel", model=squad_model, checkpoint_dir=flags.FLAGS.model_dir
  )

  model_name = flags.FLAGS.triton_model_name

  model_folder = model_export_path + "/triton_models/" + model_name
  version_folder = model_folder + "/" + str(flags.FLAGS.triton_model_version)
  final_model_folder = version_folder + "/model.savedmodel"

  if not os.path.exists(version_folder):
    os.makedirs(version_folder)
  if not os.path.exists(final_model_folder):
    os.rename(model_export_path + "/savedmodel", final_model_folder)
    print("Model saved to dir", final_model_folder)
  else:
    if flags.FLAGS.triton_model_overwrite:
      shutil.rmtree(final_model_folder)
      os.rename(model_export_path + "/savedmodel", final_model_folder)
      print("WARNING: Existing model was overwritten. Model dir: {}".format(final_model_folder))
    else:
      print(
        "ERROR: Could not save Triton model. Folder already exists. Use '--triton_model_overwrite=True' if you would like to overwrite an existing model. Model dir: {}".format(
          final_model_folder
        )
      )
      return

  config_filename = os.path.join(model_folder, "config.pbtxt")
  if os.path.exists(config_filename) and not flags.FLAGS.triton_model_overwrite:
    print(
      "ERROR: Could not save Triton model config. Config file already exists. Use '--triton_model_overwrite=True' if you would like to overwrite an existing model config. Model config: {}".format(
        config_filename
      )
    )
    return

  config_template = r"""
name: "{model_name}"
platform: "tensorflow_savedmodel"
max_batch_size: {max_batch_size}
input [
    {{
        name: "input_mask"
        data_type: TYPE_INT32
        dims: {seq_length}
    }},
    {{
        name: "input_type_ids"
        data_type: TYPE_INT32
        dims: {seq_length}
    }},
    {{
        name: "input_word_ids"
        data_type: TYPE_INT32
        dims: {seq_length}
    }}
    ]
    output [
    {{
        name: "end_positions"
        data_type: TYPE_FP32
        dims: {seq_length}
    }},
    {{
        name: "start_positions"
        data_type: TYPE_FP32
        dims: {seq_length}
    }}
]
{dynamic_batching}
instance_group [
    {{
        count: {engine_count}
        kind: KIND_GPU
        gpus: [{gpu_list}]
    }}
]"""

  batching_str = ""
  max_batch_size = flags.FLAGS.triton_max_batch_size

  if flags.FLAGS.triton_dyn_batching_delay > 0:
    # Use only full and half full batches
    pref_batch_size = [int(max_batch_size / 2.0), max_batch_size]

    batching_str = r"""
dynamic_batching {{
    preferred_batch_size: [{0}]
    max_queue_delay_microseconds: {1}
}}""".format(", ".join([str(x) for x in pref_batch_size]), int(flags.FLAGS.triton_dyn_batching_delay * 1000.0))

  config_values = {
    "model_name": model_name,
    "max_batch_size": max_batch_size,
    "seq_length": input_meta_data["max_seq_length"],
    "dynamic_batching": batching_str,
    "gpu_list": ", ".join([x.name.split(":")[-1] for x in tf.config.list_physical_devices("GPU")]),
    "engine_count": flags.FLAGS.triton_engine_count,
  }

  with open(model_folder + "/config.pbtxt", "w") as file:
    final_config_str = config_template.format_map(config_values)
    file.write(final_config_str)


def main():
  define_flags()
  with tf.io.gfile.GFile(flags.FLAGS.input_meta_data_path, "rb") as reader:
    input_meta_data = json.loads(reader.read().decode("utf-8"))

  if flags.FLAGS.mode == "export_only":
    export_squad(flags.FLAGS.model_export_path, input_meta_data)
    return

  strategy = distribution_utils.get_distribution_strategy(
    distribution_strategy=flags.FLAGS.distribution_strategy, num_gpus=flags.FLAGS.num_gpus, tpu_address=flags.FLAGS.tpu
  )

  if flags.FLAGS.use_fp16:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

  os.makedirs(flags.FLAGS.model_dir, exist_ok=True)

  if flags.FLAGS.mode in ("train", "train_and_predict"):
    train_squad(strategy, input_meta_data)
  if flags.FLAGS.mode in ("predict", "sm_predict", "trt_predict", "train_and_predict") and (
    not flags.FLAGS.use_horovod or hvd.rank() == 0
  ):
    predict_squad(strategy, input_meta_data)


if __name__ == "__main__":
  dp.runner(main)
