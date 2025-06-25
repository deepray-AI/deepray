import os

import numpy as np
import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline


class Movielens1MRating(DataPipeline):
  def __init__(self, split=False, **kwargs):
    super().__init__(**kwargs)
    self.split = split
    flags.FLAGS([
      "--feature_map={}".format(os.path.join(os.path.dirname(__file__), "movielens.csv")),
    ])
    import tensorflow_datasets as tfds

    # Ratings data.
    self.ratings = tfds.load("movielens/1m-ratings", split="train", data_dir="/datasets/", download=True)
    # Features of all the available movies.
    self.movies = tfds.load("movielens/1m-movies", split="train", data_dir="/datasets/", download=True)
    users = self.ratings.map(lambda x: x["user_id"], os.cpu_count())
    movie_ids = self.movies.map(lambda x: x["movie_id"], os.cpu_count())
    movies = self.movies.map(lambda x: x["movie_title"], os.cpu_count())
    self.user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    self.user_ids_vocabulary.adapt(users.batch(1_000_000))
    self.movie_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    self.movie_ids_vocabulary.adapt(movie_ids.batch(1_000_000))
    self.movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    self.movie_titles_vocabulary.adapt(movies.batch(1_682))

  def get_vocabulary(self, feature):
    if feature == "user_id":
      return self.user_ids_vocabulary.get_vocabulary()
    elif feature == "movie_id":
      return self.movie_ids_vocabulary.get_vocabulary()
    elif feature == "movie_title":
      return self.movie_titles_vocabulary.get_vocabulary()
    else:
      column = (
        self.original_dataset.map(lambda x: {feature: x[feature]}, os.cpu_count())
        .batch(self.__len__)
        .map(lambda x: x[feature], os.cpu_count())
      )
      return np.unique(np.concatenate(list(column)))

  def parser(self, record):
    return {
      "movie_id": self.movie_ids_vocabulary(record["movie_id"]),
      "movie_title": self.movie_titles_vocabulary(record["movie_title"]),
      "user_id": self.user_ids_vocabulary(record["user_id"]),
      "movie_genres": tf.cast(record["movie_genres"][0], tf.int32),
      "user_gender": tf.cast(record["user_gender"], tf.int32),
      "user_occupation_label": tf.cast(record["user_occupation_label"], tf.int32),
      "bucketized_user_age": tf.cast(record["bucketized_user_age"], tf.int32),
      "timestamp": tf.cast(record["timestamp"] - 880000000, tf.int32),
    }, record["user_rating"]

  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    dataset = self.ratings.map(self.parser, os.cpu_count())
    if epochs > 1:
      dataset = dataset.repeat(epochs)
    if shuffle:
      dataset = dataset.shuffle(1_000_000, seed=2021, reshuffle_each_iteration=False)
    if self.split:
      if is_training:
        dataset = dataset.take(80_000)
      else:
        dataset = dataset.skip(80_000).take(20_000)
    dataset = dataset.batch(batch_size)
    return dataset

  @property
  def __len__(self):
    return 1_000_224
