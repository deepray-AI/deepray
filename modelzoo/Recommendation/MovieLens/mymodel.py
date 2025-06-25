import tensorflow as tf
from typing import Dict, Tuple
from deepray.layers.embedding_variable import EmbeddingVariable
import deepray.custom_ops.sparse_operation_kit as sok

USE_EV = False
USE_SOK = False


class MovieLensRankingModel(tf.keras.Model):
  def __init__(self, user_input_dim=None, movie_input_dim=None, embedding_dimension=64):
    super().__init__()
    if USE_SOK:
      self.user_embed = sok.DynamicVariable(
        dimension=64,
        var_type="hybrid",
        # key_dtype=tf.int64,
        # value_dtype=tf.float32,
        # initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1./math.sqrt(emb_dim)),
        initializer="uniform",
        name="DynamicVariable_user",
        init_capacity=1024,
        max_capacity=1024,
      )
      self.movie_embed = sok.DynamicVariable(
        dimension=64,
        var_type="hybrid",
        # key_dtype=tf.int64,
        # value_dtype=tf.float32,
        # initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1./math.sqrt(emb_dim)),
        initializer="uniform",
        name="DynamicVariable_movie",
        init_capacity=1024,
        max_capacity=1024,
      )
    else:
      if USE_EV:
        self.user_embed = EmbeddingVariable(embedding_dim=embedding_dimension)
        self.movie_embed = EmbeddingVariable(embedding_dim=embedding_dimension)
      else:
        self.user_embed = tf.keras.layers.Embedding(user_input_dim, embedding_dimension)
        self.movie_embed = tf.keras.layers.Embedding(movie_input_dim, embedding_dimension)

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    # Define how the ranking scores are computed:
    # Take the dot-product of the user embeddings with the movie embeddings.
    user_id = features["user_id"]
    movie_id = features["movie_title"]
    if USE_SOK:
      user_embeddings = sok.lookup_sparse(self.user_embed, user_id)
      movie_embeddings = sok.lookup_sparse(self.movie_embed, movie_id)
    else:
      user_embeddings = self.user_embed(user_id)
      movie_embeddings = self.movie_embed(movie_id)

    return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)
