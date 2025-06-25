from ._ranking import _RankingMetric


class HitsMetric(_RankingMetric):
  r"""Hits@k metric.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  Hits@k(y, s) = 1.0, if \exists i s.t. y_i >= 1 and rank(s_i) <= k
  Hits@k(y, s) = 0.0, otherwise.
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1` and `y_i = 0` if `y_i < 1`.
  NOTE: While `topn` could be left as `None` without raising an error, the Hits
  metric without `topn` specified would be trivial as it simply measures the
  percentage of lists with at least 1 relevant item.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> hits_at_1 = tfr.keras.metrics.HitsMetric(topn=1)
  >>> hits_at_1(y_true, y_pred).numpy()
  0.0
  >>> hits_at_2 = tfr.keras.metrics.HitsMetric(topn=2)
  >>> hits_at_2(y_true, y_pred).numpy()
  1.0

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> hits_at_1 = tfr.keras.metrics.HitsMetric(topn=1, ragged=True)
  >>> hits_at_1(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.HitsMetric(topn=1)])
  ```

  Definition:

  $$
  \text{Hits}@k(\{y\}, \{s\}) = \max_{i | y_i \geq 1}
                                \mathbf{I} [\text{rank}(s_i) \leq k]
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly and $y_i$ are labels.
  """

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(HitsMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.HitsMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    config = super(HitsMetric, self).get_config()
    config.update({
      "topn": self._topn,
    })
    return config
