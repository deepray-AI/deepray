"""NCF model input pipeline."""

import functools

import tensorflow as tf

from deepray.datasets.datapipeline import DataPipeline
from deepray.datasets.movielens import constants as rconst


class Movielens(DataPipeline):
  @staticmethod
  def parser(self, serialized_data, batch_size=None, is_training=True):
    """Convert serialized TFRecords into tensors.

    Args:
      serialized_data: A tensor containing serialized records.
      batch_size: The data arrives pre-batched, so batch size is needed to
        deserialize the data.
      is_training: Boolean, whether data to deserialize to training data or
        evaluation data.
    """

    def _get_feature_map(batch_size, is_training=True):
      """Returns data format of the serialized tf record file."""

      if is_training:
        return {
          rconst.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
          rconst.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
          rconst.VALID_POINT_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
          "labels": tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
        }
      else:
        return {
          rconst.USER_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
          rconst.ITEM_COLUMN: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
          rconst.DUPLICATE_MASK: tf.io.FixedLenFeature([batch_size, 1], dtype=tf.int64),
        }

    features = tf.io.parse_single_example(serialized_data, _get_feature_map(batch_size, is_training=is_training))
    users = tf.cast(features[rconst.USER_COLUMN], rconst.USER_DTYPE)
    items = tf.cast(features[rconst.ITEM_COLUMN], rconst.ITEM_DTYPE)

    if is_training:
      valid_point_mask = tf.cast(features[rconst.VALID_POINT_MASK], tf.bool)
      fake_dup_mask = tf.zeros_like(users)
      return {
        rconst.USER_COLUMN: users,
        rconst.ITEM_COLUMN: items,
        rconst.VALID_POINT_MASK: valid_point_mask,
        rconst.TRAIN_LABEL_KEY: tf.reshape(tf.cast(features["labels"], tf.bool), (batch_size, 1)),
        rconst.DUPLICATE_MASK: fake_dup_mask,
      }
    else:
      labels = tf.cast(tf.zeros_like(users), tf.bool)
      fake_valid_pt_mask = tf.cast(tf.zeros_like(users), tf.bool)
      return {
        rconst.USER_COLUMN: users,
        rconst.ITEM_COLUMN: items,
        rconst.DUPLICATE_MASK: tf.cast(features[rconst.DUPLICATE_MASK], tf.bool),
        rconst.VALID_POINT_MASK: fake_valid_pt_mask,
        rconst.TRAIN_LABEL_KEY: labels,
      }

  def build_dataset(self, input_file_pattern, pre_batch_size, batch_size, is_training=True, rebatch=False):
    """Creates dataset from (tf)records files for training/evaluation."""
    if pre_batch_size != batch_size:
      raise ValueError("Pre-batch ({}) size is not equal to batch size ({})".format(pre_batch_size, batch_size))

    files = tf.data.Dataset.list_files(input_file_pattern, shuffle=is_training)

    dataset = files.interleave(
      tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    decode_fn = functools.partial(self.parser, batch_size=pre_batch_size, is_training=is_training)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if rebatch:
      # A workaround for TPU Pod evaluation dataset.
      # TODO (b/162341937) remove once it's fixed.
      dataset = dataset.unbatch()
      dataset = dataset.batch(pre_batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
