from abc import ABC
import tensorflow as tf

from ..tfrecord_pipeline import TFRecordPipeline


class Wikitext103(TFRecordPipeline, ABC):
  def __init__(self, bin_sizes, tgt_len, **kwargs):
    super().__init__(**kwargs)
    self.bin_sizes = bin_sizes
    self.tgt_len = tgt_len

  def parser(self, record):
    # preprocess "inp_perm" and "tgt_perm"
    def _process_perm_feature(example, prefix):
      for b in range(len(self.bin_sizes)):
        cnt = example.pop("{}_cnt_{}".format(prefix, b))[0]
        tup = example.pop("{}_tup_{}".format(prefix, b))

        tup = tf.reshape(tf.sparse_tensor_to_dense(tup), shape=[cnt, 2])

        # tf.float32
        perm = tf.sparse_to_dense(
          sparse_indices=tup, output_shape=[self.tgt_len, self.bin_sizes[b]], sparse_values=1.0, default_value=0.0
        )

        example["{}_perm_{}".format(prefix, b)] = perm

    # whether allow the last batch with a potentially shorter length
    record_spec = {
      "inputs": tf.io.VarLenFeature(tf.int64),
      "labels": tf.io.VarLenFeature(tf.int64),
    }

    # retrieve serialized example
    example = tf.io.parse_single_example(serialized=record, features=record_spec)

    # cast int64 into int32
    # cast sparse to dense
    for key in list(example.keys()):
      val = example[key]
      if tf.keras.backend.is_sparse(val):
        val = tf.sparse.to_dense(val)
      if val.dtype == tf.int64:
        val = tf.cast(val, tf.int32)
      example[key] = val

    return example["inputs"], example["labels"]
