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
"""Common functionalities used by both Keras and Estimator implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from deepray.core.common import distribution_utils

from deepray.datasets.movielens import constants as rconst
from deepray.datasets.movielens import process
from deepray.datasets.movielens import data_pipeline
from deepray.datasets.movielens import data_preprocessing
from deepray.utils.flags import core as flags_core


def get_inputs(params):
  """Returns some parameters used by the model."""
  if FLAGS.download_if_missing and not FLAGS.use_synthetic_data:
    process.download(FLAGS.dataset, FLAGS.data_dir)

  if FLAGS.random_seed is not None:
    np.random.seed(FLAGS.random_seed)

  if FLAGS.use_synthetic_data:
    producer = data_pipeline.DummyConstructor()
    num_users, num_items = rconst.DATASET_TO_NUM_USERS_AND_ITEMS[FLAGS.dataset]
    num_train_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
    num_eval_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
  else:
    num_users, num_items, producer = data_preprocessing.instantiate_pipeline(
      dataset=FLAGS.dataset,
      data_dir=FLAGS.data_dir,
      params=params,
      constructor_type=FLAGS.constructor_type,
      deterministic=FLAGS.random_seed is not None,
    )
    num_train_steps = producer.train_batches_per_epoch
    num_eval_steps = producer.eval_batches_per_epoch

  return num_users, num_items, num_train_steps, num_eval_steps, producer


def get_v1_distribution_strategy(params):
  """Returns the distribution strategy to use."""
  if params["use_tpu"]:
    # Some of the networking libraries are quite chatty.
    for name in ["googleapiclient.discovery", "googleapiclient.discovery_cache", "oauth2client.transport"]:
      logging.getLogger(name).setLevel(logging.ERROR)

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=params["tpu"], zone=params["tpu_zone"], project=params["tpu_gcp_project"], coordinator_name="coordinator"
    )

    logging.info("Issuing reset command to TPU to ensure a clean state.")
    tf.Session.reset(tpu_cluster_resolver.get_master())

    # Estimator looks at the master it connects to for MonitoredTrainingSession
    # by reading the `TF_CONFIG` environment variable, and the coordinator
    # is used by StreamingFilesDataset.
    tf_config_env = {
      "session_master": tpu_cluster_resolver.get_master(),
      "eval_session_master": tpu_cluster_resolver.get_master(),
      "coordinator": tpu_cluster_resolver.cluster_spec().as_dict()["coordinator"],
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config_env)

    distribution = tf.distribute.TPUStrategy(tpu_cluster_resolver, steps_per_run=100)

  else:
    distribution = distribution_utils.get_distribution_strategy(num_gpus=params["num_gpus"])

  return distribution


def convert_to_softmax_logits(logits):
  """Convert the logits returned by the base model to softmax logits.

  Args:
    logits: used to create softmax.

  Returns:
    Softmax with the first column of zeros is equivalent to sigmoid.
  """
  softmax_logits = tf.concat([logits * 0, logits], axis=1)
  return softmax_logits
