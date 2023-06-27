"""NCF model input pipeline."""
import os
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeLine

FLAGS = flags.FLAGS
FLAGS([sys.argv[0],
       "--num_train_examples=1000224",
       "--feature_map={}".format(os.path.join(os.path.dirname(__file__), "movielens.csv")),
       ])


class Movielens1MRating(DataPipeLine):

  def build_dataset(self, input_file_pattern,
                    batch_size,
                    is_training=True,
                    context: tf.distribute.InputContext = None,
                    use_horovod=False,
                    *args, **kwargs):
    import tensorflow_datasets as tfds
    dataset = tfds.load(input_file_pattern, split='train')
    features = dataset.map(
      lambda x: {
        "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
        "user_id": tf.strings.to_number(x["user_id"], tf.int64),
      })
    ratings = dataset.map(
      lambda x: tf.one_hot(tf.cast(x['user_rating'] - 1, dtype=tf.int64), 5))
    dataset = dataset.zip((features, ratings))
    dataset = dataset.repeat(FLAGS.epochs)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)

    return dataset
