# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for embedding layer."""

import numpy as np
import tensorflow as tf
from deepray.layers.embedding import Embedding

import keras
from keras.mixed_precision import policy
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


class EmbeddingTest(test_combinations.TestCase):
  @test_combinations.run_all_keras_modes
  def test_embedding_correctness(self):
    layer = Embedding(embedding_dim=2, vocabulary_size=2)
    model = keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 1], [2, 2]])])
    model.run_eagerly = test_utils.should_run_eagerly()
    outputs = model.predict(np.array([[0, 1, 0]], dtype="int32"))
    self.assertAllClose(outputs, [[[1, 1], [2, 2], [1, 1]]])

  def test_embedding_incorrect_dimension(self):
    with self.assertRaises(ValueError):
      Embedding(vocabulary_size=0, embedding_dim=1)

    with self.assertRaises(ValueError):
      Embedding(vocabulary_size=1, embedding_dim=0)

  @test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
  def test_eager_gpu_cpu(self):
    l = Embedding(embedding_dim=2, vocabulary_size=2)
    l.build((None, 2))
    inputs = keras.backend.constant([[0, 1, 0]], dtype="int32")
    with tf.GradientTape() as tape:
      output = l(inputs)
    gs = tape.gradient(output, l.weights)
    opt = tf.keras.optimizers.Adagrad(0.1)
    opt.apply_gradients(zip(gs, l.weights))
    self.assertAllEqual(len(gs), 1)

  @test_combinations.run_all_keras_modes
  def test_embedding_with_ragged_input(self):
    layer = Embedding(
      vocabulary_size=3,
      embedding_dim=2,
      weights=[np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])],
    )
    inputs = keras.layers.Input(shape=(None,), dtype=tf.float32, ragged=True)

    outputs = keras.layers.Lambda(lambda args: keras.backend.identity(args))(inputs)

    outputs = layer(outputs)

    model = keras.Model(inputs, outputs)
    model.run_eagerly = test_utils.should_run_eagerly()
    outputs = model.predict(tf.ragged.constant([[1.0, 2.0, 2.0], [0.0], [1.0, 2.0]], ragged_rank=1))
    self.assertAllClose(
      outputs,
      tf.ragged.constant(
        [
          [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
          [[0.0, 0.0]],
          [[1.0, 1.0], [2.0, 2.0]],
        ],
        ragged_rank=1,
      ),
    )

  @test_utils.enable_v2_dtype_behavior
  def test_mixed_precision_embedding(self):
    try:
      policy.set_global_policy("mixed_float16")
      layer = Embedding(vocabulary_size=5, embedding_dim=2)
      self.assertEqual(layer._dtype_policy.name, "mixed_float16")
      outputs = layer(np.array([0, 1, 2]))
      self.assertEqual(outputs.dtype, "float16")
    finally:
      policy.set_global_policy("float32")

  @test_combinations.run_all_keras_modes
  def test_embedding_with_sparse_input_sparse_output(self):
    layer = Embedding(
      vocabulary_size=3,
      embedding_dim=2,
      weights=[np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])],
      sparse=True,
    )
    input = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 2], dense_shape=[3, 3])
    output = layer(input)
    expected_output = tf.SparseTensor(
      indices=[[0, 1, 0], [0, 1, 1], [1, 2, 0], [1, 2, 1]],
      values=[1.0, 1.0, 2.0, 2.0],
      dense_shape=[3, 3, 2],
    )
    self.assertAllClose(output.indices, expected_output.indices)
    self.assertAllClose(output.values, expected_output.values)
    self.assertAllClose(output.dense_shape, expected_output.dense_shape)

  @test_combinations.run_all_keras_modes
  def test_embedding_with_sparse_input_dense_output(self):
    layer = Embedding(
      vocabulary_size=3,
      embedding_dim=2,
      weights=[np.array([[0.1, 0.1], [1.0, 1.0], [2.0, 2.0]])],
      sparse=False,
    )
    input = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 2], dense_shape=[3, 3])
    output = layer(input)
    expected_output = tf.constant([
      [[0.1, 0.1], [1.0, 1.0], [0.1, 0.1]],
      [[0.1, 0.1], [0.1, 0.1], [2.0, 2.0]],
      [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
    ])
    self.assertAllClose(output, expected_output)

  @test_combinations.run_all_keras_modes
  def test_embedding_with_dense_input_sprase_output(self):
    layer = Embedding(
      vocabulary_size=3,
      embedding_dim=2,
      weights=[np.array([[0, 0], [1.0, 1.0], [2.0, 2.0]])],
      sparse=True,
      mask_zero=False,
    )
    inputs = tf.constant([0, 0, 0, 2, 1])
    output = layer(inputs)
    expected_output = tf.SparseTensor(
      indices=[[3, 0], [3, 1], [4, 0], [4, 1]],
      values=[2.0, 2.0, 1.0, 1.0],
      dense_shape=[5, 2],
    )
    self.assertAllClose(output.indices, expected_output.indices)
    self.assertAllClose(output.values, expected_output.values)
    self.assertAllClose(output.dense_shape, expected_output.dense_shape)

  @test_combinations.run_all_keras_modes(always_skip_v1=True)
  def test_use_one_hot(self):
    batch = 8
    input_length = 10
    layer = Embedding(vocabulary_size=100, embedding_dim=16)
    self.assertFalse(layer._use_one_hot_matmul)

    inputs = tf.random.uniform(shape=[batch, input_length], minval=0, maxval=9, dtype=tf.int64)
    output_1 = layer(inputs)

    layer._use_one_hot_matmul = True
    output_2 = layer(inputs)

    self.assertAllClose(output_1, output_2)
    self.assertEqual(output_1.dtype, output_2.dtype)

    # Make sure the layer can be created with hidden kwargs, and not
    # serialize it into config (for now).
    layer = Embedding(vocabulary_size=100, embedding_dim=16, use_one_hot_matmul=True)
    self.assertTrue(layer._use_one_hot_matmul)

    self.assertNotIn("use_one_hot_matmul", layer.get_config())


if __name__ == "__main__":
  tf.test.main()
