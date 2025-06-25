import tensorflow as tf
from deepray.datasets.movielens import constants as rconst
from deepray.models import neumf_model


def metric_fn(logits, dup_mask, match_mlperf):
  dup_mask = tf.cast(dup_mask, tf.float32)
  logits = tf.slice(logits, [0, 1], [-1, -1])
  in_top_k, _, metric_weights, _ = neumf_model.compute_top_k_and_ndcg(logits, dup_mask, match_mlperf)
  metric_weights = tf.cast(metric_weights, tf.float32)
  return in_top_k, metric_weights


class MetricLayer(tf.keras.layers.Layer):
  """Custom layer of metrics for NCF model."""

  def __init__(self, match_mlperf):
    super(MetricLayer, self).__init__()
    self.match_mlperf = match_mlperf

  def get_config(self):
    return {"match_mlperf": self.match_mlperf}

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def call(self, inputs, training=False):
    logits, dup_mask = inputs

    if training:
      hr_sum = 0.0
      hr_count = 0.0
    else:
      metric, metric_weights = metric_fn(logits, dup_mask, self.match_mlperf)
      hr_sum = tf.reduce_sum(metric * metric_weights)
      hr_count = tf.reduce_sum(metric_weights)

    self.add_metric(hr_sum, name="hr_sum", aggregation="mean")
    self.add_metric(hr_count, name="hr_count", aggregation="mean")
    return logits


class LossLayer(tf.keras.layers.Layer):
  """Pass-through loss layer for NCF model."""

  def __init__(self, loss_normalization_factor):
    # The loss may overflow in float16, so we use float32 instead.
    super(LossLayer, self).__init__(dtype="float32")
    self.loss_normalization_factor = loss_normalization_factor
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")

  def get_config(self):
    return {"loss_normalization_factor": self.loss_normalization_factor}

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def call(self, inputs):
    logits, labels, valid_pt_mask_input = inputs
    loss = self.loss(y_true=labels, y_pred=logits, sample_weight=valid_pt_mask_input)
    loss = loss * (1.0 / self.loss_normalization_factor)
    self.add_loss(loss)
    return logits


class NCFModel(tf.keras.Model):
  def __init__(self, params, *args, **kwargs):
    super(NCFModel, self).__init__(*args, **kwargs)
    self._params = params
    self.construct_model = neumf_model.ConstructModel(params)

  def call(self, inputs, training=None, mask=None):
    user_input = inputs[rconst.USER_COLUMN]
    item_input = inputs[rconst.ITEM_COLUMN]
    dup_mask_input = inputs[rconst.DUPLICATE_MASK]
    label_input = inputs[rconst.TRAIN_LABEL_KEY]
    valid_pt_mask_input = inputs[rconst.VALID_POINT_MASK]

    logits = self.construct_model(user_input, item_input)

    zeros = tf.keras.layers.Lambda(lambda x: x * 0)(logits)

    softmax_logits = tf.keras.layers.concatenate([zeros, logits], axis=-1)

    # Custom training loop calculates loss and metric as a part of
    # training/evaluation step function.
    if not self._params["use_custom_training_loop"]:
      softmax_logits = MetricLayer(self._params["match_mlperf"])([softmax_logits, dup_mask_input])
      # TODO(b/134744680): Use model.add_loss() instead once the API is well
      # supported.
      softmax_logits = LossLayer(self._params["batch_size"])([softmax_logits, label_input, valid_pt_mask_input])
    return softmax_logits
