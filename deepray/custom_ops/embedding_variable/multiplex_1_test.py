# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multiplex_1."""

import numpy as np
import tensorflow as tf

from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import errors_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.with_eager_op_as_function
class MultiplexOpRank1Test(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    print(gen_kv_variable_ops)
    print(dir(gen_kv_variable_ops))

  # @test_util.run_in_graph_and_eager_modes
  # def test_multiplex_int(self):
  #   shape = [3]
  #   dtype = tf.float32
  #   shared_name = "var_1_2"
  #   name = "var_1/"
  #   _invalid_key_type = tf.int64
  #   container = ""
  #   gen_kv_variable_ops.kv_var_handle_op(shape=shape, dtype=dtype,
  #                                               shared_name=shared_name,
  #                                               name=name,
  #                                               Tkeys=_invalid_key_type,
  #                                               container=container)


if __name__ == '__main__':
  tf.test.main()
