import tensorflow as tf

from deepray.metrics import metrics_impl


class _RankingMetric(tf.keras.metrics.Mean):
  """Implements base ranking metric class.

  Please see tf.keras.metrics.Mean for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(_RankingMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    # An instance of `metrics_impl._RankingMetric`.
    # Overwrite this in subclasses.
    self._metric = None
    self._ragged = ragged

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)

    # TODO: Add mask argument for metric.compute() call
    per_list_metric_val, per_list_metric_weights = self._metric.compute(y_true, y_pred, sample_weight)
    return super(_RankingMetric, self).update_state(per_list_metric_val, sample_weight=per_list_metric_weights)

  def get_config(self):
    config = super(_RankingMetric, self).get_config()
    config.update({
      "ragged": self._ragged,
    })
    return config


class MRRMetric(_RankingMetric):
  r"""Mean reciprocal rank (MRR).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  MRR(y, s) = max_i y_i / rank(s_i)
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> mrr = dp.metrics.MRRMetric()
  >>> mrr(y_true, y_pred).numpy()
  0.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> mrr = dp.metrics.MRRMetric(ragged=True)
  >>> mrr(y_true, y_pred).numpy()
  0.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.MRRMetric()])
  ```

  Definition:

  $$
  \text{MRR}(\{y\}, \{s\}) = \max_i \frac{\bar{y}_i}{\text{rank}(s_i)}
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly and $\bar{y_i}$ are truncated labels:

  $$
  \bar{y}_i = \begin{cases}
  1 & \text{if }y_i \geq 1 \\
  0 & \text{else}
  \end{cases}
  $$
  """

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(MRRMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MRRMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    config = super(MRRMetric, self).get_config()
    config.update({
      "topn": self._topn,
    })
    return config
