from deepray.models.framework import FrameWork
from deepray.layers.fm import FactorizationMachine
from deepray.layers.mlp import MLP


class DeepFM(FrameWork):
  def __init__(
    self,
    wide_column=None,
    fm_column=None,
    deep_column=None,
    dnn_hidden_units=[1024, 256, 32],
    final_hidden_units=[128, 64],
    optimizer_type="adam",
    learning_rate=0.001,
    inputs=None,
    use_bn=True,
    bf16=False,
    stock_tf=None,
    adaptive_emb=False,
    input_layer_partitioner=None,
    dense_layer_partitioner=None,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._deep_net = MLP(hidden_units=dnn_hidden_units)
    self._final_deep_net = MLP(hidden_units=final_hidden_units)
    self._fm_net = FactorizationMachine()
    self._logit_layer = tf.keras.layers.Dense(1)

  def build_features(self):
    # input features
    with tf.variable_scope("input_layer", partitioner=self._input_layer_partitioner, reuse=tf.AUTO_REUSE):
      fm_cols = {}
      if self._adaptive_emb and not self.tf:
        """Adaptive Embedding Feature Part 1 of 2"""
        adaptive_mask_tensors = {}
        for col in CATEGORICAL_COLUMNS:
          adaptive_mask_tensors[col] = tf.ones([args.batch_size], tf.int32)
        dnn_input = tf.feature_column.input_layer(
          features=self._feature, feature_columns=self._deep_column, adaptive_mask_tensors=adaptive_mask_tensors
        )
        wide_input = tf.feature_column.input_layer(
          self._feature, self._wide_column, cols_to_output_tensors=fm_cols, adaptive_mask_tensors=adaptive_mask_tensors
        )
      else:
        dnn_input = tf.feature_column.input_layer(self._feature, self._deep_column)
        wide_input = tf.feature_column.input_layer(self._feature, self._wide_column, cols_to_output_tensors=fm_cols)

      fm_input = tf.stack([fm_cols[cols] for cols in self._fm_column], 1)

    return wide_input, fm_input, dnn_input

  def build_network(self, flags=None, features=None):
    wide_input, fm_input, dnn_input = self.build_features()
    dnn_output = self._deep_net(dnn_input)

    # linear / fisrt order part
    linear_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)

    # FM second order part
    fm_output = self._fm_net(fm_input)

    # Final dnn layer
    all_input = tf.concat([dnn_output, linear_output, fm_output], 1)

    dnn_logits = self._final_deep_net(all_input)

    _logits = self._logit_layer(dnn_logits)
    probability = tf.math.sigmoid(_logits)
    output = tf.round(probability)
    return output
