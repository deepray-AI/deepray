# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Skip-gram sampling ops from https://arxiv.org/abs/1301.3781."""

import csv
import tensorflow as tf

from deepray.utils.resource_loader import LazySO

from deepray.utils.types import AcceptableDTypes, FloatTensorLike, TensorLike
from typing import Optional

_skip_gram_so = LazySO("custom_ops/text/_skip_gram_ops.so")

tf.no_gradient("Deepray>SkipGramGenerateCandidates")


def skip_gram_sample(
  input_tensor: TensorLike,
  min_skips: FloatTensorLike = 1,
  max_skips: FloatTensorLike = 5,
  start: FloatTensorLike = 0,
  limit: FloatTensorLike = -1,
  emit_self_as_target: bool = False,
  vocab_freq_table: tf.lookup.KeyValueTensorInitializer = None,
  vocab_min_count: Optional[FloatTensorLike] = None,
  vocab_subsampling: Optional[FloatTensorLike] = None,
  corpus_size: Optional[FloatTensorLike] = None,
  seed: Optional[FloatTensorLike] = None,
  name: Optional[str] = None,
) -> tf.Tensor:
  """Generates skip-gram token and label paired Tensors from the input
  tensor.

  Generates skip-gram `("token", "label")` pairs using each element in the
  rank-1 `input_tensor` as a token. The window size used for each token will
  be randomly selected from the range specified by `[min_skips, max_skips]`,
  inclusive. See https://arxiv.org/abs/1301.3781 for more details about
  skip-gram.

  For example, given `input_tensor = ["the", "quick", "brown", "fox",
  "jumps"]`, `min_skips = 1`, `max_skips = 2`, `emit_self_as_target = False`,
  the output `(tokens, labels)` pairs for the token "quick" will be randomly
  selected from either `(tokens=["quick", "quick"], labels=["the", "brown"])`
  for 1 skip, or `(tokens=["quick", "quick", "quick"],
  labels=["the", "brown", "fox"])` for 2 skips.

  If `emit_self_as_target = True`, each token will also be emitted as a label
  for itself. From the previous example, the output will be either
  `(tokens=["quick", "quick", "quick"], labels=["the", "quick", "brown"])`
  for 1 skip, or `(tokens=["quick", "quick", "quick", "quick"],
  labels=["the", "quick", "brown", "fox"])` for 2 skips.

  The same process is repeated for each element of `input_tensor` and
  concatenated together into the two output rank-1 `Tensors` (one for all the
  tokens, another for all the labels).

  If `vocab_freq_table` is specified, tokens in `input_tensor` that are not
  present in the vocabulary are discarded. Tokens whose frequency counts are
  below `vocab_min_count` are also discarded. Tokens whose frequency
  proportions in the corpus exceed `vocab_subsampling` may be randomly
  down-sampled. See Eq. 5 in http://arxiv.org/abs/1310.4546 for more details
  about subsampling.

  Args:
    input_tensor: A rank-1 `Tensor` from which to generate skip-gram
      candidates.
    min_skips: `int` or scalar `Tensor` specifying the minimum window size to
      randomly use for each token. Must be >= 0 and <= `max_skips`. If
      `min_skips` and `max_skips` are both 0, the only label outputted will
      be the token itself when `emit_self_as_target = True` -
      or no output otherwise.
    max_skips: `int` or scalar `Tensor` specifying the maximum window size to
      randomly use for each token. Must be >= 0.
    start: `int` or scalar `Tensor` specifying the position in
      `input_tensor` from which to start generating skip-gram candidates.
    limit: `int` or scalar `Tensor` specifying the maximum number of
      elements in `input_tensor` to use in generating skip-gram candidates.
      -1 means to use the rest of the `Tensor` after `start`.
    emit_self_as_target: `bool` or scalar `Tensor` specifying whether to emit
      each token as a label for itself.
    vocab_freq_table: (Optional) A lookup table (subclass of
      `lookup.InitializableLookupTableBase`) that maps tokens to their raw
      frequency counts. If specified, any token in `input_tensor` that is not
      found in `vocab_freq_table` will be filtered out before generating
      skip-gram candidates. While this will typically map to integer raw
      frequency counts, it could also map to float frequency proportions.
      `vocab_min_count` and `corpus_size` should be in the same units
      as this.
    vocab_min_count: (Optional) `int`, `float`, or scalar `Tensor` specifying
      minimum frequency threshold (from `vocab_freq_table`) for a token to be
      kept in `input_tensor`. If this is specified, `vocab_freq_table` must
      also be specified - and they should both be in the same units.
    vocab_subsampling: (Optional) `float` specifying frequency proportion
      threshold for tokens from `input_tensor`. Tokens that occur more
      frequently (based on the ratio of the token's `vocab_freq_table` value
      to the `corpus_size`) will be randomly down-sampled. Reasonable
      starting values may be around 1e-3 or 1e-5. If this is specified, both
      `vocab_freq_table` and `corpus_size` must also be specified. See Eq. 5
      in http://arxiv.org/abs/1310.4546 for more details.
    corpus_size: (Optional) `int`, `float`, or scalar `Tensor` specifying the
      total number of tokens in the corpus (e.g., sum of all the frequency
      counts of `vocab_freq_table`). Used with `vocab_subsampling` for
      down-sampling frequently occurring tokens. If this is specified,
      `vocab_freq_table` and `vocab_subsampling` must also be specified.
    seed: (Optional) `int` used to create a random seed for window size and
      subsampling. See `set_random_seed` docs for behavior.
    name: (Optional) A `string` name or a name scope for the operations.

  Returns:
    A `tuple` containing (token, label) `Tensors`. Each output `Tensor` is of
    rank-1 and has the same type as `input_tensor`.

  Raises:
    ValueError: If `vocab_freq_table` is not provided, but `vocab_min_count`,
      `vocab_subsampling`, or `corpus_size` is specified.
      If `vocab_subsampling` and `corpus_size` are not both present or
      both absent.
  """

  if vocab_freq_table is None and (
    vocab_min_count is not None or vocab_subsampling is not None or corpus_size is not None
  ):
    raise ValueError(
      "vocab_freq_table is not provided, but vocab_min_count={}, "
      "vocab_subsampling={}, or corpus_size={} is not None."
      "These settings are useless without a vocab_freq_table.".format(vocab_min_count, vocab_subsampling, corpus_size)
    )

  if (vocab_subsampling is None) != (corpus_size is None):
    raise ValueError(
      "vocab_subsampling is {} while corpus_size is {} - both must be "
      "provided in order for subsampling to work.".format(vocab_subsampling, corpus_size)
    )

  with tf.name_scope(name or "skip_gram_sample"):
    input_tensor = _filter_input(
      input_tensor=input_tensor,
      vocab_freq_table=vocab_freq_table,
      vocab_min_count=vocab_min_count,
      vocab_subsampling=vocab_subsampling,
      corpus_size=corpus_size,
      seed=seed,
    )

    seed1, seed2 = tf.compat.v1.get_seed(seed)
    tokens, labels = _skip_gram_so.ops.deepray_skip_gram_generate_candidates(
      input_tensor=input_tensor,
      min_skips=min_skips,
      max_skips=max_skips,
      start=start,
      limit=limit,
      emit_self_as_target=emit_self_as_target,
      # Note that seed here should be seed1! This is due to
      # GuardedPhiloxRandom's hard-coded attributes of "seed" and "seed2".
      seed=seed1,
      seed2=seed2,
    )

    # TODO(weiho): If the need arises, add support for sparse input_tensor that
    # figures out sentence boundaries, then calls
    # skip_gram_generate_candidates() on each sentence.

    return tokens, labels


def skip_gram_sample_with_text_vocab(
  input_tensor: TensorLike,
  vocab_freq_file: str,
  vocab_token_index: FloatTensorLike = 0,
  vocab_token_dtype: Optional[AcceptableDTypes] = tf.dtypes.string,
  vocab_freq_index: FloatTensorLike = 1,
  vocab_freq_dtype: Optional[AcceptableDTypes] = tf.dtypes.float64,
  vocab_delimiter: str = ",",
  vocab_min_count: Optional[FloatTensorLike] = None,
  vocab_subsampling: Optional[FloatTensorLike] = None,
  corpus_size: Optional[FloatTensorLike] = None,
  min_skips: FloatTensorLike = 1,
  max_skips: FloatTensorLike = 5,
  start: FloatTensorLike = 0,
  limit: FloatTensorLike = -1,
  emit_self_as_target: bool = False,
  seed: Optional[FloatTensorLike] = None,
  name: Optional[str] = None,
) -> tf.Tensor:
  """Skip-gram sampling with a text vocabulary file.

  Wrapper around `skip_gram_sample()` for use with a text vocabulary file.
  The vocabulary file is expected to be a plain-text file, with lines of
  `vocab_delimiter`-separated columns. The `vocab_token_index` column should
  contain the vocabulary term, while the `vocab_freq_index` column should
  contain the number of times that term occurs in the corpus. For example,
  with a text vocabulary file of:

    ```
    bonjour,fr,42
    hello,en,777
    hola,es,99
    ```

  You should set `vocab_delimiter=","`, `vocab_token_index=0`, and
  `vocab_freq_index=2`.

  See `skip_gram_sample()` documentation for more details about the skip-gram
  sampling process.

  Args:
    input_tensor:
      A rank-1 `Tensor` from which to generate skip-gram candidates.
    vocab_freq_file:
      `string` specifying full file path to the text vocab file.
    vocab_token_index: `int` specifying which column in the text vocab file
      contains the tokens.
    vocab_token_dtype:
      `DType` specifying the format of the tokens in the text vocab file.
    vocab_freq_index: `int` specifying which column in the text vocab file
      contains the frequency counts of the tokens.
    vocab_freq_dtype: `DType` specifying the format of the frequency counts
      in the text vocab file.
    vocab_delimiter: `string` specifying the delimiter used in the text vocab
      file.
    vocab_min_count: `int`, `float`, or scalar `Tensor` specifying
      minimum frequency threshold (from `vocab_freq_file`) for a token to be
      kept in `input_tensor`. This should correspond with `vocab_freq_dtype`.
    vocab_subsampling: (Optional) `float` specifying frequency proportion
      threshold for tokens from `input_tensor`. Tokens that occur more
      frequently will be randomly down-sampled. Reasonable starting values
      may be around 1e-3 or 1e-5. See Eq. 5 in http://arxiv.org/abs/1310.4546
      for more details.
    corpus_size: (Optional) `int`, `float`, or scalar `Tensor` specifying the
      total number of tokens in the corpus (e.g., sum of all the frequency
      counts of `vocab_freq_file`). Used with `vocab_subsampling` for
      down-sampling frequently occurring tokens. If this is specified,
      `vocab_freq_file` and `vocab_subsampling` must also be specified.
      If `corpus_size` is needed but not supplied, then it will be calculated
      from `vocab_freq_file`. You might want to supply your own value if you
      have already eliminated infrequent tokens from your vocabulary files
      (where frequency < vocab_min_count) to save memory in the internal
      token lookup table. Otherwise, the unused tokens' variables will waste
      memory.  The user-supplied `corpus_size` value must be greater than or
      equal to the sum of all the frequency counts of `vocab_freq_file`.
    min_skips: `int` or scalar `Tensor` specifying the minimum window size to
      randomly use for each token. Must be >= 0 and <= `max_skips`. If
      `min_skips` and `max_skips` are both 0, the only label outputted will
      be the token itself.
    max_skips: `int` or scalar `Tensor` specifying the maximum window size to
      randomly use for each token. Must be >= 0.
    start: `int` or scalar `Tensor` specifying the position in `input_tensor`
      from which to start generating skip-gram candidates.
    limit: `int` or scalar `Tensor` specifying the maximum number of elements
      in `input_tensor` to use in generating skip-gram candidates. -1 means
      to use the rest of the `Tensor` after `start`.
    emit_self_as_target: `bool` or scalar `Tensor` specifying whether to emit
      each token as a label for itself.
    seed: (Optional) `int` used to create a random seed for window size and
      subsampling. See
      [`set_random_seed`](../../g3doc/python/constant_op.md#set_random_seed)
      for behavior.
    name: (Optional) A `string` name or a name scope for the operations.

  Returns:
    A `tuple` containing (token, label) `Tensors`. Each output `Tensor` is of
    rank-1 and has the same type as `input_tensor`.

  Raises:
    ValueError: If `vocab_token_index` or `vocab_freq_index` is less than 0
      or exceeds the number of columns in `vocab_freq_file`.
      If `vocab_token_index` and `vocab_freq_index` are both set to the same
      column. If any token in `vocab_freq_file` has a negative frequency.
  """

  if vocab_token_index < 0 or vocab_freq_index < 0:
    raise ValueError(
      "vocab_token_index={} and vocab_freq_index={} must both be >= 0.".format(vocab_token_index, vocab_freq_index)
    )
  if vocab_token_index == vocab_freq_index:
    raise ValueError(
      "vocab_token_index and vocab_freq_index should be different, but are both {}.".format(vocab_token_index)
    )

  # Iterates through the vocab file and calculates the number of vocab terms as
  # well as the total corpus size (by summing the frequency counts of all the
  # vocab terms).
  calculated_corpus_size = 0.0
  vocab_size = 0
  with tf.io.gfile.GFile(vocab_freq_file, mode="r") as f:
    reader = csv.reader(f, delimiter=vocab_delimiter)
    for row in reader:
      if vocab_token_index >= len(row) or vocab_freq_index >= len(row):
        raise ValueError(
          "Row in vocab file only has {} columns, "
          "so vocab_token_index={} or "
          "vocab_freq_index={} is out of bounds. Row content: {}".format(
            len(row), vocab_token_index, vocab_freq_index, row
          )
        )
      vocab_size += 1
      freq = vocab_freq_dtype.as_numpy_dtype(row[vocab_freq_index])
      if freq < 0:
        raise ValueError("Row in vocab file has negative frequency of {}. Row content: {}".format(freq, row))
      # Note: tokens whose frequencies are below vocab_min_count will still
      # contribute to the total corpus size used for vocab subsampling.
      calculated_corpus_size += freq

  if not corpus_size:
    corpus_size = calculated_corpus_size
  elif calculated_corpus_size - corpus_size > 1e-6:
    raise ValueError(
      "`corpus_size`={} must be greater than or equal to the "
      "sum of all the frequency counts ({}) of `vocab_freq_file` ({}).".format(
        corpus_size, calculated_corpus_size, vocab_freq_file
      )
    )

  vocab_freq_table = tf.lookup.StaticHashTable(
    tf.lookup.TextFileInitializer(
      filename=vocab_freq_file,
      key_dtype=vocab_token_dtype,
      key_index=vocab_token_index,
      value_dtype=vocab_freq_dtype,
      value_index=vocab_freq_index,
      vocab_size=vocab_size,
      delimiter=vocab_delimiter,
    ),
    # For vocab terms not in vocab file, use a default value of -1.
    default_value=-1,
  )

  return skip_gram_sample(
    input_tensor,
    min_skips=min_skips,
    max_skips=max_skips,
    start=start,
    limit=limit,
    emit_self_as_target=emit_self_as_target,
    vocab_freq_table=vocab_freq_table,
    vocab_min_count=vocab_min_count,
    vocab_subsampling=vocab_subsampling,
    # corpus_size is not used unless vocab_subsampling is specified.
    corpus_size=None if vocab_subsampling is None else corpus_size,
    seed=seed,
    name=name,
  )


def _filter_input(
  input_tensor,
  vocab_freq_table,
  vocab_min_count,
  vocab_subsampling,
  corpus_size,
  seed,
):
  input_tensor = tf.convert_to_tensor(input_tensor)
  """Filters input tensor based on vocab freq, threshold, and subsampling."""
  if vocab_freq_table is None:
    return input_tensor

  if not isinstance(vocab_freq_table, tf.lookup.StaticHashTable):
    raise ValueError(
      "vocab_freq_table must be a subclass of "
      "InitializableLookupTableBase (such as HashTable) instead of type "
      "{}.".format(type(vocab_freq_table))
    )

  with tf.name_scope("filter_vocab"):
    freq = vocab_freq_table.lookup(input_tensor)
    # Filters out elements in input_tensor that are not found in
    # vocab_freq_table (table returns a default value of -1 specified above when
    # an element is not found).
    mask = tf.math.not_equal(freq, vocab_freq_table.default_value)

    # Filters out elements whose vocab frequencies are less than the threshold.
    if vocab_min_count is not None:
      cast_threshold = tf.cast(vocab_min_count, freq.dtype)
      mask = tf.math.logical_and(mask, tf.math.greater_equal(freq, cast_threshold))

    input_tensor = tf.boolean_mask(input_tensor, mask)
    freq = tf.boolean_mask(freq, mask)

  if not vocab_subsampling:
    return input_tensor

  if vocab_subsampling < 0 or vocab_subsampling > 1:
    raise ValueError("Invalid vocab_subsampling={} - it should be within range [0, 1].".format(vocab_subsampling))

  # Subsamples the input tokens based on vocabulary frequency and
  # vocab_subsampling threshold (ie randomly discard commonly appearing
  # tokens).
  with tf.name_scope("subsample_vocab"):
    corpus_size = tf.cast(corpus_size, tf.dtypes.float64)
    freq = tf.cast(freq, tf.dtypes.float64)
    vocab_subsampling = tf.cast(vocab_subsampling, tf.dtypes.float64)

    # From tensorflow_models/tutorials/embedding/word2vec_kernels.cc, which is
    # suppose to correlate with Eq. 5 in http://arxiv.org/abs/1310.4546.
    keep_prob = (tf.math.sqrt(freq / (vocab_subsampling * corpus_size)) + 1.0) * (
      vocab_subsampling * corpus_size / freq
    )
    random_prob = tf.random.uniform(tf.shape(freq), minval=0, maxval=1, dtype=tf.dtypes.float64, seed=seed)

    mask = tf.math.less_equal(random_prob, keep_prob)
    return tf.boolean_mask(input_tensor, mask)
