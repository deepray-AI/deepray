import tensorflow as tf
from absl import flags
from tensorflow.keras.layers import Dense

from deepray.layers.dynamic_embedding import DistributedDynamicEmbedding
from deepray.utils.data.feature_map import FeatureMap

FLAGS = flags.FLAGS


class Demo(tf.keras.Model):

  def __init__(self, embedding_size, is_training=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if is_training:
      initializer = tf.keras.initializers.VarianceScaling()
    else:
      initializer = tf.keras.initializers.Zeros()
    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map
    self.features_dict = {}
    for key, dtype, emb_size, length in self.feature_map.loc[self.feature_map["ftype"] == "Categorical"][[
        "name", "dtype", "dim", "length"
    ]].values:
      self.features_dict[key] = DistributedDynamicEmbedding(
          embedding_dim=emb_size,
          key_dtype=dtype,
          value_dtype=tf.float32,
          initializer=initializer,
          device="HBM",
          name=key + '_DynamicEmbeddingLayer',
      )
    self.d0 = Dense(
        256,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )
    self.d1 = Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )
    self.d2 = Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    )

  def call(self, features, *args, **kwargs):

    movie_id = features["movie_id"]
    user_id = features["user_id"]

    user_id_weights = self.features_dict['user_id'](user_id)

    movie_id_weights = self.features_dict['movie_id'](movie_id)

    embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)

    dnn = self.d0(embeddings)
    dnn = self.d1(dnn)
    dnn = self.d2(dnn)
    out = tf.reshape(dnn, shape=[-1])

    return out
    # loss = tf.keras.losses.MeanSquaredError()(rating, out)
    # predictions = {"out": out}
