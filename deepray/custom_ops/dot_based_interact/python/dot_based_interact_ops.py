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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

from deepray.utils.resource_loader import LazySO

_dot_based_interact_ops = LazySO("custom_ops/dot_based_interact/_dot_based_interact_ops.so")

dot_based_interact = _dot_based_interact_ops.ops.dot_based_interact


@ops.RegisterGradient("DotBasedInteract")
def dot_based_interact_grad(op, grad):
  input = op.inputs[0]
  return _dot_based_interact_ops.ops.dot_based_interact_grad(input, grad)
