# Copyright 2023 The Deepray Authors. All Rights Reserved.
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
import argparse
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys

import tensorflow as tf
from absl import flags

# Local project imports
from deepray import activations
from deepray import callbacks
from deepray import custom_ops
from deepray import layers
from deepray import losses
from deepray import metrics
from deepray import models
from deepray import optimizers
from deepray import options
from deepray.register import register_all
from deepray.utils import logging_util
from deepray.utils import types
from deepray.utils.ensure_tf_install import _check_tf_version
from deepray.utils.flags import common_flags
from deepray.version import __version__

# _check_tf_version()

logger = logging_util.get_logger()

common_flags.define_common_flags()
flags.FLAGS(sys.argv, known_only=True)


def init():
  logger.debug(f"sys.argv = {sys.argv}")  # sys.argv from Horovod

  gpus = tf.config.experimental.list_physical_devices("GPU")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  if flags.FLAGS.distribution_strategy == "horovod":
    import horovod.tensorflow as hvd

    hvd.init()
    if gpus:
      from deepray.utils import gpu_affinity

      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
      gpu_affinity.set_affinity(hvd.local_rank())


def start_tensorflow_server(cluster_resolver):
  # Set the environment variable to allow reporting worker and ps failure to the
  # coordinator. This is a workaround and won't be necessary in the future.
  os.environ["GRPC_FAIL_FAST"] = "use_caller"

  server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True,
  )
  server.join()


def runner(function, verbose=None):
  parser = argparse.ArgumentParser(description="Deepray Runner")
  parser.add_argument("-v", "--version", action="version", version=__version__, help="Shows Deepray version.")
  parser.add_argument(
    "--distribution_strategy", type=str, default="Horovod", help="Whether run distributed training with Horovod."
  )

  physical_devices = tf.config.list_physical_devices("GPU")
  world_size = len(physical_devices)
  logger.debug(f"world_size = {world_size}")

  user_argv = sys.argv  # get user specified args
  args, unknown = parser.parse_known_args()

  if world_size > 1 and args.distribution_strategy == "Horovod":
    user_argv.extend([
      "--distribution_strategy=horovod",
      f"--num_gpus={world_size}",
      "--use_horovod",
    ])
    try:
      import horovod

      os.environ["HOROVOD_STALL_CHECK_TIME_SECONDS"] = "5"
      os.environ["HOROVOD_STALL_SHUTDOWN_TIME_SECONDS"] = "30"
    except ImportError:
      raise ValueError("Please install Horovod properly first if you want to use Horovod distribution_strategy.")

    def helper(argv, main):
      logger.debug(f"argv = {argv}")
      init()
      main()

    horovod.run(helper, args=(sys.argv,), kwargs={"main": function}, np=world_size, verbose=verbose, use_mpi=True)
  elif args.distribution_strategy == "ParameterServer":
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
      start_tensorflow_server(cluster_resolver)
    else:
      user_argv.extend(["--distribution_strategy=parameter_server"])
      init()
      function()
  else:
    logger.info("Deepray finds only one GPU available, so we turn off distribution_strategy.")
    user_argv.extend(["--distribution_strategy=off", f"--num_gpus={world_size}"])
    init()
    function()
