import os
from typing import Dict, Tuple
import tensorflow as tf
from mymodel import MovieLensRankingModel, USE_SOK

import deepray as dp
import deepray.custom_ops.sparse_operation_kit as sok
from deepray.optimizers.multi_optimizer import MultiOptimizer
from deepray.datasets.movielens.movielens_100k_ratings import Movielens100kRating


class TestData(Movielens100kRating):

  def parser(self, x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    labels = x.pop("user_rating")
    x["movie_title"] = self.movie_titles_vocabulary(x["movie_title"])
    x["user_id"] = self.user_ids_vocabulary(x["user_id"])
    x["movie_id"] = self.movie_ids_vocabulary(x["movie_id"])
    return x, labels

  def build_dataset(
      self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    key_func = lambda x: self.user_ids_vocabulary(x["user_id"])
    reduce_func = lambda key, dataset: dataset.batch(100)

    ratings = self.ratings.map(
        lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "movie_id": x["movie_id"],
            "user_rating": x["user_rating"]
        }
    )
    ds_train = ratings.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)
    ds_train = ds_train.map(self.parser)

    # ds_train = ds_train.ragged_batch(batch_size=batch_size)
    ds_train = ds_train.ragged_batch(batch_size=batch_size)
    return ds_train


data_pipe = TestData()
ds_train = data_pipe(128)

for x, label in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:3, :3].numpy()}")
    print()
  print(f"Shape of label: {label.shape}")
  print(f"Example values of label: {label[:3, :3].numpy()}")

# Create the ranking model, trained with a ranking loss and evaluated with
# ranking metrics.
model = MovieLensRankingModel(
    user_input_dim=data_pipe.user_ids_vocabulary.vocabulary_size(),
    movie_input_dim=data_pipe.movie_titles_vocabulary.vocabulary_size()
)

if USE_SOK:
  embedding_opt = dp.optimizers.Adagrad(learning_rate=0.5)
  embedding_opt = sok.OptimizerWrapper(embedding_opt)
  dense_opt = dp.optimizers.Adagrad(learning_rate=0.5)
  optimizer = MultiOptimizer([
      (embedding_opt, "DynamicVariable_"),
  ], default_optimizer=dense_opt)
else:
  optimizer = dp.optimizers.Adagrad(0.5)

loss = dp.losses.SoftmaxLoss(ragged=True)
eval_metrics = [
    dp.metrics.NDCGMetric(ragged=True, name="metric/ndcg"),
    dp.metrics.MRRMetric(ragged=True, name="metric/mrr"),
]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics, run_eagerly=False)

model.fit(ds_train, epochs=3)

# Get movie title candidate list.
movies = data_pipe.movies.map(lambda x: x["movie_title"], os.cpu_count())
for movie_titles in movies.batch(2000):
  break

# Generate the input for user 42.
inputs = {
    "user_id": tf.expand_dims(tf.repeat(data_pipe.user_ids_vocabulary("42"), repeats=movie_titles.shape[0]), axis=0),
    "movie_title": tf.expand_dims(data_pipe.movie_titles_vocabulary(movie_titles), axis=0)
}

# Get movie recommendations for user 42.
scores = model(inputs)
titles = dp.metrics.utils.sort_by_scores(scores, [tf.expand_dims(movie_titles, axis=0)])[0]
print(f"Top 5 recommendations for user 42: {titles[0, :5]}")
