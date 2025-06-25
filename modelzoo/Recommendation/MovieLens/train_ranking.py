from typing import Dict

import tensorflow as tf
from absl import flags

import deepray as dp
from deepray.core.trainer import Trainer
from deepray.datasets.movielens.movielens_100k_ratings import Movielens100kRating
from deepray.layers.embedding_variable import EmbeddingVariable


class RankingModel(tf.keras.Model):

  def __init__(self, embedding_dimension=32):
    super().__init__()
    # Compute embeddings for users.
    self.user_embeddings = EmbeddingVariable(embedding_dim=embedding_dimension)
    self.movie_embeddings = EmbeddingVariable(embedding_dim=embedding_dimension)

    # Compute predictions.
    self.ratings = tf.keras.Sequential(
        [
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ]
    )

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    user_id, movie_title = inputs["user_id"], inputs["movie_title"]
    user_id = tf.reshape(user_id, [-1])
    movie_title = tf.reshape(movie_title, [-1])
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)
    emb_vec = tf.concat([user_embedding, movie_embedding], axis=1)
    return self.ratings(emb_vec)


data_pipe = Movielens100kRating(split=True)
train_dataset = data_pipe(flags.FLAGS.batch_size, is_training=True)
test_dataset = data_pipe(flags.FLAGS.batch_size, is_training=True)

optimizer = dp.optimizers.Adagrad(0.1)
model = RankingModel()

score = model(
    {
        "user_id": data_pipe.user_ids_vocabulary("42"),
        "movie_title": data_pipe.movie_titles_vocabulary("One Flew Over the Cuckoo's Nest (1975)")
    }
)
print(score)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])


cached_train = train_dataset.cache()
cached_test = test_dataset.cache()

trainer.fit(x=cached_train)

trainer.evaluate(cached_test, return_dict=True)

test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model(
      {
          "user_id": data_pipe.user_ids_vocabulary("42"),
          "movie_title": data_pipe.movie_titles_vocabulary(movie_title)
      }
  )

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")
