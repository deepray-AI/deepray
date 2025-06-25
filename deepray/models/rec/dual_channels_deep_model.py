import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de


class DualChannelsDeepModel(tf.keras.Model):
  def __init__(self, user_embedding_size=1, movie_embedding_size=1, embedding_initializer=None, is_training=True):
    if not is_training:
      de.enable_inference_mode()

    super(DualChannelsDeepModel, self).__init__()
    self.user_embedding_size = user_embedding_size
    self.movie_embedding_size = movie_embedding_size

    if embedding_initializer is None:
      embedding_initializer = tf.keras.initializers.Zeros()

    self.user_embedding = de.keras.layers.SquashedEmbedding(
      user_embedding_size, initializer=embedding_initializer, name="user_embedding"
    )
    self.movie_embedding = de.keras.layers.SquashedEmbedding(
      movie_embedding_size, initializer=embedding_initializer, name="movie_embedding"
    )

    self.dnn1 = tf.keras.layers.Dense(
      64,
      activation="relu",
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
    )
    self.dnn2 = tf.keras.layers.Dense(
      16,
      activation="relu",
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
    )
    self.dnn3 = tf.keras.layers.Dense(
      5,
      activation="softmax",
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
    )
    self.bias_net = tf.keras.layers.Dense(
      5,
      activation="softmax",
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
    )

  @tf.function
  def call(self, features):
    user_id = tf.reshape(features["user_id"], (-1, 1))
    movie_id = tf.reshape(features["movie_id"], (-1, 1))
    user_latent = self.user_embedding(user_id)
    movie_latent = self.movie_embedding(movie_id)
    latent = tf.concat([user_latent, movie_latent], axis=1)

    x = self.dnn1(latent)
    x = self.dnn2(x)
    x = self.dnn3(x)

    bias = self.bias_net(latent)
    x = 0.2 * x + 0.8 * bias
    return x
