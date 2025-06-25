from ._ranking import _RankingMetric


class ARPMetric(_RankingMetric):
  r"""Average relevance position (ARP).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  ARP(y, s) = sum_i (y_i * rank(s_i)) / sum_j y_j
  ```

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> arp = tfr.keras.metrics.ARPMetric()
  >>> arp(y_true, y_pred).numpy()
  2.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> arp = tfr.keras.metrics.ARPMetric(ragged=True)
  >>> arp(y_true, y_pred).numpy()
  1.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.ARPMetric()])
  ```

  Definition:

  $$
  \text{ARP}(\{y\}, \{s\}) =
  \frac{1}{\sum_i y_i} \sum_i y_i \cdot \text{rank}(s_i)
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly.
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(ARPMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._metric = metrics_impl.ARPMetric(name=name, ragged=ragged)
