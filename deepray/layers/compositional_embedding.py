# -*- coding:utf-8 -*-
"""Compositional Embedding layer."""

import tensorflow as tf
from tensorflow.python.framework import dtypes

from deepray.layers.embedding_variable import EmbeddingVariable


class CompositionalEmbedding(EmbeddingVariable):
  """
  Compositional Embedding is designed for reducing Large-scale Sparse Embedding Weights.
  See:
  [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems](https://arxiv.org/abs/1909.02107)
  [Binary Code based Hash Embedding for Web-scale Applications](https://arxiv.org/pdf/2109.02471)
  """

  STRATEGIES = {"Q-R"}
  OPERATIONS = {"add", "mul", "concat"}
  SUPPORTED_DTYPES = {"int32", "int64"}

  def __init__(
    self,
    composition_size: int,
    key_dtype=dtypes.int64,
    complementary_strategy: str = "Q-R",
    operation: str = "add",
    name: str = "compositional_embedding",
    **kwargs,
  ):
    super(CompositionalEmbedding, self).__init__(name=f"{name}/Compositional", key_dtype=key_dtype, **kwargs)
    if complementary_strategy not in self.STRATEGIES:
      raise ValueError(
        f"Strategy '{complementary_strategy}' is not supported. Available strategies: {list(self.STRATEGIES)}"
      )
    if operation not in self.OPERATIONS:
      raise ValueError(f"Operation '{operation}' is not supported. Available operations: {list(self.OPERATIONS)}")
    # if complementary_strategy == 'Q-R':
    #   if num_of_partitions != 2:
    #     raise ValueError("the num_of_partitions must be 2 when using Q-R strategy.")

    self.key_dtype = key_dtype
    self.composition_factor = self.factor2decimal(composition_size)
    self.complementary_strategy = complementary_strategy
    self.operation = operation

  def factor2decimal(self, composition_part: int):
    if self.key_dtype == "int32":
      base = 32
    elif self.key_dtype == "int64":
      base = 64
    else:
      raise ValueError(f"{self.key_dtype} type not support yet.")

    # Calculate the quotient and remainder of A divided by composition_size.
    quotient = base // composition_part
    remainder = base % composition_part

    # Create a list of length composition_size with each element equal to the quotient.
    result = [quotient] * composition_part

    # Distribute the remainder among the first few elements of the list.
    for i in range(remainder):
      result[i] += 1

    # Sort the list in ascending order.
    result.sort()

    res = []
    for i in range(len(result)):
      binary_str = ""
      for j in range(len(result)):
        binary_str += result[j] * ("1" if i == j else "0")

      int_num = int(binary_str, 2) - 2**base if int(binary_str[0]) else int(binary_str, 2)
      res.append(int_num)
    return res

  def read(self, ids, *args, **kwargs):
    ids_QRP = [tf.bitwise.bitwise_and(ids, x) for x in self.composition_factor]
    results = tf.split(
      self.embedding_variable.sparse_read(ids_QRP), num_or_size_splits=len(self.composition_factor), axis=0
    )
    new_result = [tf.squeeze(x, axis=0) for x in results]
    if self.operation == "add":
      ret = tf.add_n(new_result)
      return ret
    elif self.operation == "mul":
      ret = tf.multiply(new_result)
      return ret
    elif self.operation == "concat":
      ret = tf.concat(new_result, 1)
      return ret
    else:
      raise ValueError(f"{self.operation} operation not support yet.")
