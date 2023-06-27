"""NCF model input pipeline."""

import os
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeLine

FLAGS = flags.FLAGS
FLAGS([sys.argv[0], "--num_train_examples=100000",
       "--feature_map={}".format(os.path.join(os.path.dirname(__file__), "movielens.csv")),
       ])


class Movielens100kRating(DataPipeLine):

  def parser(self, record):
    return {
      "movie_id": tf.strings.to_number(record["movie_id"], tf.int64),
      "user_id": tf.strings.to_number(record["user_id"], tf.int64),
      "movie_genres": tf.cast(record["movie_genres"][0], tf.int32),
      "user_gender": tf.cast(record["user_gender"], tf.int32),
      "user_occupation_label": tf.cast(record["user_occupation_label"], tf.int32),
      "raw_user_age": tf.cast(record["raw_user_age"], tf.int32),
      "timestamp": tf.cast(record["timestamp"] - 880000000, tf.int32),
    }, record["user_rating"]

  def build_dataset(self, input_file_pattern,
                    batch_size,
                    is_training=True,
                    context: tf.distribute.InputContext = None,
                    use_horovod=False,
                    *args, **kwargs):
    import tensorflow_datasets as tfds
    ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings = ratings.map(self.parser
                          # lambda x: {
                          #   "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
                          #   "user_id": tf.strings.to_number(x["user_id"], tf.int64),
                          #   "user_rating": x["user_rating"]
                          # }
                          )
    ratings = ratings.repeat(FLAGS.epochs)
    shuffled = ratings.shuffle(1_000_000,
                               seed=2021,
                               reshuffle_each_iteration=False)
    dataset = shuffled.batch(batch_size)
    return dataset
