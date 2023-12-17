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
"""Tests for ffm ops."""

import tensorflow as tf
from tensorflow.python.platform import test

from deepray.custom_ops.ffm_ops import ffm


class FFMOpsTest(test.TestCase):

  def _test_ffm_mul(self, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output = ffm(left=left, right=right, dim_size=4)
      self.assertTrue(output.shape == (8, 480))

  def _test_ffm_mul_grad(self, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4)
        loss = tf.reduce_sum(out)
        left_grad, right_grad = g.gradient(loss, [left, right])
        self.assertTrue(left_grad.shape == (8, 40))
        self.assertTrue(right_grad.shape == (8, 48))

  def _test_ffm_dot(self, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output = ffm(left=left, right=right, dim_size=4, int_type='dot')
      self.assertTrue(output.shape == (8, 120))

  def _test_ffm_dot_grad(self, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4, int_type='dot')
        loss = tf.reduce_sum(out)
        left_grad, right_grad = g.gradient(loss, [left, right])

        self.assertTrue(left_grad.shape == (8, 40))
        self.assertTrue(right_grad.shape == (8, 48))

  def _testCpu(self):
    self._test_ffm_mul(use_gpu=False)
    self._test_ffm_mul_grad(use_gpu=False)
    self._test_ffm_dot(use_gpu=False)
    self._test_ffm_dot_grad(use_gpu=False)

  def _testGpu(self):
    self._test_ffm_mul(use_gpu=True)
    self._test_ffm_mul_grad(use_gpu=True)
    self._test_ffm_dot(use_gpu=True)
    self._test_ffm_dot_grad(use_gpu=True)

  def testAll(self):
    self._testCpu()
    self._testGpu()


if __name__ == "__main__":
  test.main()
