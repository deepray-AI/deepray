# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

# We don't want the whole process to quit because of the import failure when
# we don't use horovod to do communication.
try:
  import horovod.tensorflow as hvd
except ImportError:
  pass
from absl import flags

from deepray.utils import logging_util

logger = logging_util.get_logger()


def get_rank():
  try:
    return hvd.rank()
  except:
    return 0


def get_world_size():
  try:
    return hvd.size()
  except:
    return 1


def is_main_process():
  return not flags.FLAGS.use_horovod or get_rank() == 0


def main_info(info):
  if is_main_process():
    logger.info(info)


def main_warning(info):
  if is_main_process():
    logger.warning(info)


def id_in_rank():
  return 0


def num_gpu_per_rank():
  return 1


def global_gpu_id():
  return get_rank()
