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


class CommToolBase(object):
  """Abstract base class for different communication tools.

  This class assumes that a process can control multiple GPUs and there can be multiple
  processes on a node(a physical machine).

  A `context`(in the comments below) corresponds to a unique GPU, a typical example is
  the context of tensorflow's mirrored strategy.

  There should only be one instance of this class(in fact, its subclass) per process.
  """

  def rank(self):
    """Returns the global id for the current `process`.

    Similar to horovod, `rank` represents the global id of the current process. The
    difference is that `rank` is not equal to the global id of GPU since we don't
    assume that there is only one GPU in a process.

    `rank` belongs to [0, num_ranks-1].
    """
    raise NotImplementedError("rank() is not implemented")

  def num_ranks(self):
    """Returns how many processes are running in total.

    This includes all process on different nodes.
    """
    raise NotImplementedError("num_ranks() is not implemented")

  def local_rank(self):
    """Returns the local id of the current process in the current node.

    Similar to `rank`, `local rank` is not equal to the GPU id in current node since
    a process may control multiple GPUs.
    """
    raise NotImplementedError("local_rank() is not implemented")

  def num_gpus(self):
    """Returns how many GPUs are running in total.

    `num_gpus` should be equal to `num_ranks * num_gpu_per_rank`.
    """
    raise NotImplementedError("num_gpus() is not implemented")

  def alltoall(self, tensor, splits):
    """Performs an alltoall operation.

    Args:
      tensor: A tensorflow tensor to be sent.
      splits: A tensorflow tensor representing how much data should be sent to each GPU.
    """
    raise NotImplementedError("alltoall() is not implemented")

  def allreduce(self, tensor, op):
    """Performs an allreduce operation.

    Args:
      tensor: A tensorflow tensor to be sent.
      op    : A python string representing the reduce type.
    """
    raise NotImplementedError("allreduce() is not implemented")

  def allgather(self, tensor):
    """Performs an allgather operation.

    Args:
      tensor: A tensorflow tensor to be sent.
    """
    raise NotImplementedError("allgather() is not implemented")

  def broadcast(self, tensor, root):
    """Performs an allgather operation.

    Args:
      tensor: A tensorflow tensor to be sent.
    """
    raise NotImplementedError("allgather() is not implemented")


class HorovodTool(CommToolBase):
  def rank(self):
    return hvd.rank()

  def num_ranks(self):
    return hvd.size()

  def local_rank(self):
    return hvd.local_rank()

  def num_gpus(self):
    return hvd.size()

  def alltoall(self, tensor, splits):
    return hvd.alltoall(tensor, splits)

  def allreduce(self, tensor, op):
    # TODO: Add more op options
    if op == "sum":
      op = hvd.Sum
    elif op == "average":
      op = hvd.Average
    elif op == "max":
      op = hvd.Max
    elif op == "min":
      op = hvd.Min
    return hvd.allreduce(tensor, op=op)

  def allgather(self, tensor):
    return hvd.allgather(tensor)

  def allgather_object(self, obj):
    return hvd.allgather_object(obj)

  def broadcast(self, tensor, root):
    return hvd.broadcast(tensor, root)

  def broadcast_object(self, tensor, root_rank=0):
    return hvd.broadcast_object(tensor, root_rank)


# The global communication tool instance, it should only be set by `set_comm_tool` method.
_COMM_TOOL = HorovodTool()


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


def main_print(info):
  if is_main_process():
    print(info)


def main_warning(info):
  if is_main_process():
    logger.warning(info)


def rank():
  return _COMM_TOOL.rank()


def num_ranks():
  return _COMM_TOOL.num_ranks()


def local_rank():
  return _COMM_TOOL.local_rank()


def num_gpus():
  return _COMM_TOOL.num_gpus()


def alltoall(*args, **kwargs):
  return _COMM_TOOL.alltoall(*args, **kwargs)


def allreduce(*args, **kwargs):
  return _COMM_TOOL.allreduce(*args, **kwargs)


def allgather(*args, **kwargs):
  return _COMM_TOOL.allgather(*args, **kwargs)


def allgather_object(*args, **kwargs):
  return _COMM_TOOL.allgather_object(*args, **kwargs)


def broadcast(*args, **kwargs):
  return _COMM_TOOL.broadcast(*args, **kwargs)


def broadcast_object(*args, **kwargs):
  return _COMM_TOOL.broadcast_object(*args, **kwargs)
