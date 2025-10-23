import os

import tensorflow as tf
from absl import flags
from tensorflow.keras import layers
from packaging.version import parse
from deepray.layers.dynamic_embedding import DistributedDynamicEmbedding
from deepray.layers.embedding_variable import EmbeddingVariable

if parse(tf.__version__.replace("-tf", "+tf")) < parse("2.11"):
  from tensorflow.keras import layers as keras_layers
  from tensorflow.keras import initializers as keras_initializers
  from tensorflow.keras import models as keras_models
else:
  from tf_keras import layers as keras_layers
  from tf_keras import initializers as keras_initializers
  from tf_keras import models as keras_models


class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    if os.getenv("USE_TF") == "1":
      self.target_embedding = layers.Embedding(
        vocab_size,
        embedding_dim,
        name="w2v_embedding",
        embeddings_initializer=keras_initializers.Constant(value=0.1),
      )
      self.context_embedding = layers.Embedding(
        vocab_size,
        embedding_dim,
        embeddings_initializer=keras_initializers.Constant(value=0.1),
      )
    elif flags.FLAGS.use_dynamic_embedding:
      self.target_embedding = DistributedDynamicEmbedding(
        embedding_dim=embedding_dim,
        name="w2v_embedding",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=keras_initializers.Constant(value=0.1),
      )
      self.context_embedding = DistributedDynamicEmbedding(
        embedding_dim=embedding_dim,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=keras_initializers.Constant(value=0.1),
      )
    else:
      self.target_embedding = EmbeddingVariable(
        embedding_dim=embedding_dim,
        name="w2v_embedding",
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        storage_type="DRAM",
        optimizer_type="Adam",
        initializer=keras_initializers.Constant(value=0.1),
      )
      self.context_embedding = EmbeddingVariable(
        embedding_dim=embedding_dim,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        storage_type="DRAM",
        optimizer_type="Adam",
        initializer=keras_initializers.Constant(value=0.1),
      )

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum("be,bce->bc", word_emb, context_emb)
    # dots: (batch, context)
    return dots
