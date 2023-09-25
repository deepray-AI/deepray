from typing import Dict

import tensorflow as tf
from absl import flags
from deepray.layers.dcn import Cross
from deepray.layers.dot_interaction import DotInteraction
from deepray.layers.dynamic_embedding import DistributedDynamicEmbedding, DynamicEmbeddingOption
from deepray.layers.mlp import MLP
from deepray.utils.data.feature_map import FeatureMap

FLAGS = flags.FLAGS


class Ranking(tf.keras.models.Model):
  """A configurable ranking model.

  This class represents a sensible and reasonably flexible configuration for a
  ranking model that can be used for tasks such as CTR prediction.

  It can be customized as needed, and its constituent blocks can be changed by
  passing user-defined alternatives.

  For example:
  - Pass
    `feature_interaction = tfrs.layers.feature_interaction.DotInteraction()`
    to train a DLRM model, or pass
    ```
    feature_interaction = tf.keras.Sequential([
      tf.keras.layers.Concatenate(),
      tfrs.layers.feature_interaction.Cross()
    ])
    ```
    to train a DCN model.
  - Pass `task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())`
    to train a CTR prediction model, and
    `tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())` to train
    a rating prediction model.

  Changing these should cover a broad range of models, but this class is not
  intended to cover all possible use cases.  For full flexibility, inherit
  from `tfrs.models.Model` and provide your own implementations of
  the `compute_loss` and `call` methods.
  """

  def __init__(self, interaction, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map

    self._bottom_stack = MLP(hidden_units=[256, 64, 16], activations=[None, None, "relu"])
    self._top_stack = MLP(hidden_units=[512, 256, 1], activations=[None, None, "sigmoid"])

    if interaction == 'dot':
      self._feature_interaction = DotInteraction(
        skip_gather=True)
    elif interaction == 'cross':
      self._feature_interaction = tf.keras.Sequential([
        tf.keras.layers.Concatenate(),
        Cross()
      ])
    else:
      raise ValueError(
        f'params.task.model.interaction {self.task_config.model.interaction} '
        f'is not supported it must be either \'dot\' or \'cross\'.')

  def build(self, input_shape):
    self._embedding_layer = {}
    for name, dim, dtype in self.feature_map[(self.feature_map['ftype'] == "Categorical")][["name", "dim", "dtype"]].values:
      self._embedding_layer[name] = DistributedDynamicEmbedding(
      embedding_dim=dim,
      key_dtype=dtype,
      value_dtype=tf.float32,
      initializer=None,
      name=name+'_emb',
      de_option=DynamicEmbeddingOption(device="HBM", init_capacity=1 * 1024 * 1024,),
    )
    

  def call(self, inputs: Dict[str, tf.Tensor], training=None, mask=None) -> tf.Tensor:
    """Executes forward and backward pass, returns loss.

    Args:
      inputs: Model function inputs (features and labels).

    Returns:
      loss: Scalar tensor.
    """
    dense_features = inputs["dense_features"]
    sparse_embeddings = []
    for name, dim in self.feature_map[(self.feature_map['ftype'] == "Categorical")][["name", "dim"]].values:
      tensor = inputs[name]
      sparse_embeddings.append(self._embedding_layer[name](tensor))

    # Combine a dictionary into a vector and squeeze dimension from
    # (batch_size, 1, emb) to (batch_size, emb).
    sparse_embeddings = tf.nest.flatten(sparse_embeddings)

    sparse_embedding_vecs = [
      tf.squeeze(sparse_embedding) for sparse_embedding in sparse_embeddings
    ]
    dense_embedding_vec = self._bottom_stack(dense_features)

    interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
    interaction_output = self._feature_interaction(interaction_args)
    feature_interaction_output = tf.concat(
      [dense_embedding_vec, interaction_output], axis=1)

    prediction = self._top_stack(feature_interaction_output)

    return tf.reshape(prediction, [-1])
