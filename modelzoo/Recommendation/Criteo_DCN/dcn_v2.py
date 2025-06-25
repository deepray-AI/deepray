from typing import Dict

import tensorflow as tf
import tf_keras as keras
from absl import flags

from deepray.custom_ops.embedding_variable import group_embedding_lookup_ops
from deepray.layers.dcn import Cross
from deepray.layers.dot_interaction import DotInteraction
from deepray.layers.dynamic_embedding import DistributedDynamicEmbedding
from deepray.layers.embedding_variable import EmbeddingVariable
from deepray.layers.mlp import MLP
from deepray.utils.data.feature_map import FeatureMap


class EmbeddingContainer(tf.Module):
  def __init__(self, training, use_group_embedding):
    super().__init__()
    self.embeddings = {}
    self.training = training
    self.use_group_embedding = use_group_embedding

  def add_embedding(self, name, dim, dtype, voc_size):
    if voc_size:
      self.embeddings[name] = keras.layers.Embedding(
        input_dim=voc_size + 1,
        output_dim=dim,
        embeddings_initializer="uniform" if self.training else keras.initializers.Zeros(),
      )
    elif flags.FLAGS.use_dynamic_embedding:
      self.embeddings[name] = DistributedDynamicEmbedding(
        embedding_dim=dim,
        key_dtype=dtype,
        value_dtype=tf.float32,
        initializer=keras.initializers.TruncatedNormal() if self.training else keras.initializers.Zeros(),
        name="DynamicVariable_" + name,
        device="DRAM",
        init_capacity=1024 * 10,
        max_capacity=1024 * 100,
      )
    else:
      emb = EmbeddingVariable(
        embedding_dim=dim,
        key_dtype=dtype,
        value_dtype=tf.float32,
        initializer=keras.initializers.TruncatedNormal() if self.training else keras.initializers.Zeros(),
        name="emb" + name,
        # storage_type="DRAM",
        # with_unique=True,
        storage_type="HBM",
      )
      self.embeddings[name] = emb

  def __call__(self, name, tensor):
    return self.embeddings[name](tensor)

  def get_embedding_list(self):
    if not self.use_group_embedding:
      return None
    emb_list = [emb for emb in self.embeddings.values() if isinstance(emb, EmbeddingVariable)]
    return emb_list


class Ranking(keras.Model):
  def __init__(self, interaction, training=True, use_group_embedding=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.feature_map = FeatureMap().feature_map
    self._bottom_stack = MLP(hidden_units=[256, 64, 16], activations=[None, None, "relu"])
    self._top_stack = MLP(hidden_units=[512, 256, 1], activations=[None, None, "sigmoid"])
    self._interaction = interaction
    self.training = training
    if interaction == "dot":
      self._feature_interaction = DotInteraction(skip_gather=True)
    elif interaction == "cross":
      self._feature_interaction = Cross()
    else:
      raise ValueError(
        f"params.task.model.interaction {self.task_config.model.interaction} "
        f"is not supported it must be either 'dot' or 'cross'."
      )
    self.use_group_embedding = use_group_embedding
    self.embedding_container = EmbeddingContainer(training, use_group_embedding)

  def build(self, input_shape):
    for name, dim, dtype, voc_size in self.feature_map[(self.feature_map["ftype"] == "Categorical")][
      ["name", "dim", "dtype", "voc_size"]
    ].values:
      self.embedding_container.add_embedding(name, dim, dtype, voc_size)

  def call(self, inputs: Dict[str, tf.Tensor], training=None, mask=None) -> tf.Tensor:
    dense_features = inputs["dense_features"]
    sparse_embedding_vecs = []
    indices = []  # Keep indices for group embedding lookup

    for name, dim in self.feature_map[(self.feature_map["ftype"] == "Categorical")][["name", "dim"]].values:
      tensor = inputs[name]
      if self.use_group_embedding:
        indices.append(tensor)
      else:
        sparse_embedding_vecs.append(self.embedding_container(name, tensor))

    if self.use_group_embedding:
      embedding_list = self.embedding_container.get_embedding_list()
      sparse_embedding_vecs = group_embedding_lookup_ops.group_embedding_lookup(embedding_list, indices)

    dense_embedding_vec = self._bottom_stack(dense_features)

    interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
    if self._interaction == "cross":
      interaction_args = tf.concat(interaction_args, axis=-1)

    interaction_output = self._feature_interaction(interaction_args)
    feature_interaction_output = tf.concat([dense_embedding_vec, interaction_output], axis=1)

    prediction = self._top_stack(feature_interaction_output)

    return tf.reshape(prediction, [-1])
