from ._ranking import _RankingMetric


class RecallMetric(_RankingMetric):
  r"""Recall@k (R@k).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  R@K(y, s) = sum_i I[rank(s_i) < k] y_i / sum_j y_j
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> recall_at_2 = tfr.keras.metrics.RecallMetric(topn=2)
  >>> recall_at_2(y_true, y_pred).numpy()
  0.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> recall_at_2 = tfr.keras.metrics.RecallMetric(topn=2, ragged=True)
  >>> recall_at_2(y_true, y_pred).numpy()
  0.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.RecallMetric()])
  ```

  Definition:

  $$
  \text{R@k}(\{y\}, \{s\}) =
  \frac{\sum_i I[\text{rank}(s_i) \leq k] \bar{y}_i}{\sum_j \bar{y}_j}
  $$

  where:

  * $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores $s$
    with ties broken randomly
  * $I[]$ is the indicator function:\
    $I[\text{cond}] = \begin{cases}
    1 & \text{if cond is true}\\
    0 & \text{else}\end{cases}
    $
  * $\bar{y}_i$ are the truncated labels:\
    $
    \bar{y}_i = \begin{cases}
    1 & \text{if }y_i \geq 1 \\
    0 & \text{else}
    \end{cases}
    $
  * $k = |y|$ if $k$ is not provided
  """

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(RecallMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.RecallMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    config = super(RecallMetric, self).get_config()
    config.update({
      "topn": self._topn,
    })
    return config
