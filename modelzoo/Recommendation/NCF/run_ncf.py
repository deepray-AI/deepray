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
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""
import json
import sys

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from deepray.models import ncf_common

from deepray.utils.flags import core as flags_core

from deepray.callbacks.custom_early_stopping import CustomEarlyStopping
from deepray.callbacks.increment_epoch import IncrementEpochCallback
from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.movielens import Produce
from deepray.metrics.hit_rate_metric import HitRateMetric

from deepray.models.ncf_model import NCFModel
from deepray.utils.misc import keras_utils

# pylint: disable=g-bad-import-order
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS
FLAGS(
    [
        sys.argv[0],
        # "--distribution_strategy=off",
        # "--run_eagerly=true",
        "--dataset=ml-1m",
        "--eval_batch_size=160000",
    ]
)


def define_ncf_flags():
  """Add flags for running ncf_main."""
  # Add common flags
  flags_core.define_base(
      train_data=False,
      model_dir=False,
      clean=False,
      epochs=False,
      epochs_between_evals=True,
      export_dir=False,
      stop_threshold=False,
  )
  flags_core.define_performance(synthetic_data=True,
                                # fp16_implementation=True,
                                # loss_scale=True,
                               )
  flags_core.define_device(tpu=True)

  flags.adopt_module_key_flags(flags_core)

  # flags_core.set_defaults(
  #   model_dir="/tmp/ncf/",
  #   data_dir="/tmp/movielens-data/",
  #   dataset=rconst.ML_1M,
  #   epochs=2,
  #   batch_size=99000,
  #   tpu=None)

  flags.DEFINE_float(
      name="beta1", default=0.9, help=flags_core.help_wrap("beta1 hyperparameter for the Adam optimizer.")
  )

  flags.DEFINE_float(
      name="beta2", default=0.999, help=flags_core.help_wrap("beta2 hyperparameter for the Adam optimizer.")
  )

  flags.DEFINE_float(
      name="epsilon", default=1e-8, help=flags_core.help_wrap("epsilon hyperparameter for the Adam "
                                                              "optimizer.")
  )

  flags.DEFINE_float(
      name="hr_threshold",
      default=1.0,
      help=flags_core.help_wrap(
          "If passed, training will stop when the evaluation metric HR is "
          "greater than or equal to hr_threshold. For dataset ml-1m, the "
          "desired hr_threshold is 0.68 which is the result from the paper; "
          "For dataset ml-20m, the threshold can be set as 0.95 which is "
          "achieved by MLPerf implementation."
      )
  )

  flags.DEFINE_enum(
      name="constructor_type",
      default="bisection",
      enum_values=["bisection", "materialized"],
      case_sensitive=False,
      help=flags_core.help_wrap(
          "Strategy to use for generating false negatives. materialized has a"
          "precompute that scales badly, but a faster per-epoch construction"
          "time and can be faster on very large systems."
      )
  )

  flags.DEFINE_string(name="train_dataset_path", default=None, help=flags_core.help_wrap("Path to training data."))

  flags.DEFINE_string(name="eval_dataset_path", default=None, help=flags_core.help_wrap("Path to evaluation data."))

  flags.DEFINE_bool(
      name="output_ml_perf_compliance_logging",
      default=False,
      help=flags_core.help_wrap(
          "If set, output the MLPerf compliance logging. This is only useful "
          "if one is running the model for MLPerf. See "
          "https://github.com/mlperf/policies/blob/master/training_rules.adoc"
          "#submission-compliance-logs for details. This uses sudo and so may "
          "ask for your password, as root access is needed to clear the system "
          "caches, which is required for MLPerf compliance."
      )
  )

  # @flags.validator(
  #   "eval_batch_size",
  #   "eval_batch_size must be at least {}".format(rconst.NUM_EVAL_NEGATIVES +
  #                                                1))
  # def eval_size_check(eval_batch_size):
  #   return (eval_batch_size is None or
  #           int(eval_batch_size) > rconst.NUM_EVAL_NEGATIVES)

  flags.DEFINE_bool(
      name="early_stopping",
      default=False,
      help=flags_core.help_wrap("If True, we stop the training when it reaches hr_threshold")
  )


def parse_flags(flags_obj):
  """Convenience function to turn flags into params."""
  num_gpus = flags_core.get_num_gpus(flags_obj)

  batch_size = flags_obj.batch_size
  eval_batch_size = flags_obj.eval_batch_size or flags_obj.batch_size

  return {
      "epochs": flags_obj.epochs,
      "batches_per_step": 1,
      "use_seed": flags_obj.random_seed is not None,
      "batch_size": batch_size,
      "eval_batch_size": eval_batch_size,
      "learning_rate": flags_obj.learning_rate,
      "mf_dim": flags_obj.num_factors,
      "model_layers": [int(layer) for layer in flags_obj.layers],
      "mf_regularization": flags_obj.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in flags_obj.mlp_regularization],
      "num_neg": flags_obj.num_neg,
      "distribution_strategy": flags_obj.distribution_strategy,
      "num_gpus": num_gpus,
      "use_tpu": flags_obj.tpu is not None,
      "tpu": flags_obj.tpu,
      "tpu_zone": flags_obj.tpu_zone,
      "tpu_gcp_project": flags_obj.tpu_gcp_project,
      "beta1": flags_obj.beta1,
      "beta2": flags_obj.beta2,
      "epsilon": flags_obj.epsilon,
      "match_mlperf": flags_obj.benchmark,
      "epochs_between_evals": flags_obj.epochs_between_evals,
      "keras_use_ctl": flags_obj.keras_use_ctl,
      "hr_threshold": flags_obj.hr_threshold,
      "stream_files": flags_obj.tpu is not None,
      "train_dataset_path": flags_obj.train_dataset_path,
      "eval_dataset_path": flags_obj.eval_dataset_path,
  }


def build_loss(y_true, y_pred, weights):
  # The loss can overflow in float16, so we cast to float32.
  softmax_logits = tf.cast(y_pred, "float32")
  loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum",
                                                       from_logits=True)(y_true, softmax_logits, sample_weight=weights)
  loss *= (1.0 / FLAGS.batch_size)
  return loss


def run_ncf(_):
  """Run NCF training and eval with Keras."""

  params = parse_flags(FLAGS)
  params["use_tpu"] = (FLAGS.distribution_strategy == "tpu")

  if params["use_tpu"] and not params["keras_use_ctl"]:
    logging.error("Custom training loop must be used when using TPUStrategy.")
    return

  batch_size = params["batch_size"]
  time_callback = keras_utils.TimeHistory(batch_size, FLAGS.log_steps)
  callbacks = [time_callback]

  producer, input_meta_data = None, None
  generate_input_online = params["train_dataset_path"] is None

  if generate_input_online:
    # Start data producing thread.
    num_users, num_items, _, _, producer = ncf_common.get_inputs(params)
    producer.start()
    per_epoch_callback = IncrementEpochCallback(producer)
    callbacks.append(per_epoch_callback)
  else:
    assert params["eval_dataset_path"] and params["input_meta_data_path"]
    with tf.io.gfile.GFile(params["input_meta_data_path"], "rb") as reader:
      input_meta_data = json.loads(reader.read().decode("utf-8"))
      num_users = input_meta_data["num_users"]
      num_items = input_meta_data["num_items"]

  params["num_users"], params["num_items"] = num_users, num_items

  if FLAGS.early_stopping:
    early_stopping_callback = CustomEarlyStopping("val_HR_METRIC", desired_value=FLAGS.hr_threshold)
    callbacks.append(early_stopping_callback)

  data_pipe = Produce(params, producer)
  train_input_dataset = data_pipe(is_training=True)
  eval_input_dataset = data_pipe(is_training=False)

  # (train_input_dataset, eval_input_dataset, num_train_steps,
  #  num_eval_steps) = ncf_input_pipeline.create_ncf_input_data(
  #   params, producer, input_meta_data, strategy)
  num_train_steps = producer.train_batches_per_epoch
  num_eval_steps = producer.eval_batches_per_epoch

  strategy = distribution_utils.get_distribution_strategy()
  with distribution_utils.get_strategy_scope(strategy):
    keras_model = NCFModel(params)

  trainer = Trainer(
      model=keras_model,
      loss=build_loss,
      metrics=[
          tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32),
          HitRateMetric(params["match_mlperf"])
      ],
      callbacks=callbacks
  )
  train_loss, eval_results = trainer.fit(
      train_input=train_input_dataset, eval_input=eval_input_dataset, eval_steps=num_eval_steps
  )

  stats = build_stats(train_loss, eval_results, time_callback)
  return stats


def build_stats(loss, eval_result, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    loss: The final loss at training time.
    eval_result: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback likely used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if loss:
    stats["loss"] = loss

  if eval_result:
    # stats["eval_loss"] = eval_result['eval_loss']
    stats["eval_hit_rate"] = eval_result['hit_rate']

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats["step_timestamp_log"] = timestamp_log
    stats["train_finish_time"] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats["avg_exp_per_second"] = (
          time_callback.batch_size * time_callback.log_steps * (len(time_callback.timestamp_log) - 1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp)
      )

  return stats


def main(_):
  logging.info("Result is %s", run_ncf(FLAGS))


if __name__ == "__main__":
  define_ncf_flags()
  app.run(main)
