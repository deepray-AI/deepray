# Copyright 2023 The TFPlus Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops to use FlashAttentions.
"""

import tensorflow as tf
from tensorflow.python.framework import ops

from tensorflow.python.platform import resource_loader

gen_flash_attention_ops = tf.load_op_library(resource_loader.get_path_to_datafile("../../_flash_attn_ops.so"))


@ops.RegisterGradient("FMHAForward")
def _FMHA_Grad(op, grad):  # pylint: disable=invalid-name
  """Gradient for fmha_forward op."""
  # Build appropriately shaped IndexedSlices
  query = op.inputs[0]  # B X S, H, K
  key = op.inputs[1]  # B X S, H, K
  value = op.inputs[2]  # B X S, H, K
  cu_seqlens_q = op.inputs[3]
  cu_seqlens_k = op.inputs[4]
  max_seqlen_q = op.inputs[5]
  max_seqlen_k = op.inputs[6]
  rng_state = op.outputs[3]

  out = op.outputs[0]  # B X S, H, K

  p_dropout = op.get_attr("p_dropout")
  softmax_scale = op.get_attr("softmax_scale")
  zero_tensor = op.get_attr("zero_tensors")
  is_causal = op.get_attr("is_causal")
  return_softmax = op.get_attr("return_softmax")
  num_splits = op.get_attr("num_splits")
  softmax_lse = op.outputs[1]

  dq, dk, dv = gen_flash_attention_ops.fmha_backward(
      query, key, value, cu_seqlens_q, cu_seqlens_k, out, grad, softmax_lse, max_seqlen_q, max_seqlen_k, rng_state,
      p_dropout, softmax_scale, zero_tensor, is_causal, return_softmax, num_splits
  )
  return [dq, dk, dv, None, None, None, None]
