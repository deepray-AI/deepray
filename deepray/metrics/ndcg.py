import tensorflow as tf
from deepray.metrics import metrics_impl
from deepray.metrics import utils

_DEFAULT_GAIN_FN = lambda label: tf.pow(2.0, label) - 1

_DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.0) / tf.math.log1p(rank)


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


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class NDCGMetric(_RankingMetric):
  r"""Normalized discounted cumulative gain (NDCG).

  Normalized discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002])
  is the normalized version of `tfr.keras.metrics.DCGMetric`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  NDCG(y, s) = DCG(y, s) / DCG(y, y)
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```

  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> ndcg = dp.metrics.NDCGMetric()
  >>> ndcg(y_true, y_pred).numpy()
  0.6934264

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> ndcg = dp.metrics.NDCGMetric(ragged=True)
  >>> ndcg(y_true, y_pred).numpy()
  0.7974351

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.NDCGMetric()])
  ```

  Definition:

  $$
  \text{NDCG}(\{y\}, \{s\}) =
  \frac{\text{DCG}(\{y\}, \{s\})}{\text{DCG}(\{y\}, \{y\})} \\
  \text{DCG}(\{y\}, \{s\}) =
  \sum_i \text{gain}(y_i) \cdot \text{rank_discount}(\text{rank}(s_i))
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly.

  References:

    - [Cumulated gain-based evaluation of IR techniques, Järvelin et al,
       2002][jarvelin2002]

  [jarvelin2002]: https://dl.acm.org/doi/10.1145/582415.582418
  """

  def __init__(self, name=None, topn=None, gain_fn=None, rank_discount_fn=None, dtype=None, ragged=False, **kwargs):
    super(NDCGMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn or utils.pow_minus_1
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._metric = metrics_impl.NDCGMetric(
      name=name, topn=topn, gain_fn=self._gain_fn, rank_discount_fn=self._rank_discount_fn, ragged=ragged
    )

  def get_config(self):
    base_config = super(NDCGMetric, self).get_config()
    config = {
      "topn": self._topn,
      "gain_fn": self._gain_fn,
      "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config
