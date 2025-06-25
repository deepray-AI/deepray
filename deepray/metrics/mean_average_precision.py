from ._ranking import _RankingMetric


class MeanAveragePrecisionMetric(_RankingMetric):
  r"""Mean average precision (MAP).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  MAP(y, s) = sum_k (P@k(y, s) * rel(k)) / sum_i y_i
  rel(k) = y_i if rank(s_i) = k
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(topn=2)
  >>> map_metric(y_true, y_pred).numpy()
  0.25

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(
  ...   topn=2, ragged=True)
  >>> map_metric(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                metrics=[tfr.keras.metrics.MeanAveragePrecisionMetric()])
  ```

  Definition:

  $$
  \text{MAP}(\{y\}, \{s\}) =
  \frac{\sum_k P@k(y, s) \cdot \text{rel}(k)}{\sum_j \bar{y}_j} \\
  \text{rel}(k) = \max_i I[\text{rank}(s_i) = k] \bar{y}_i
  $$

  where:

  * $P@k(y, s)$ is the Precision at rank $k$. See
    `tfr.keras.metrics.PrecisionMetric`.
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
  """

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(MeanAveragePrecisionMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MeanAveragePrecisionMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    base_config = super(MeanAveragePrecisionMetric, self).get_config()
    config = {
      "topn": self._topn,
    }
    config.update(base_config)
    return config
