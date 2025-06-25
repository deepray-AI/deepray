from ._ranking import _RankingMetric


class OPAMetric(_RankingMetric):
  r"""Ordered pair accuracy (OPA).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  OPA(y, s) = sum_i sum_j I[s_i > s_j] I[y_i > y_j] / sum_i sum_j I[y_i > y_j]
  ```

  NOTE: Pairs with equal labels (`y_i = y_j`) are always ignored. Pairs with
  equal scores (`s_i = s_j`) are considered incorrectly ordered.

  Standalone usage:

  >>> y_true = [[0., 1., 2.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> opa = tfr.keras.metrics.OPAMetric()
  >>> opa(y_true, y_pred).numpy()
  0.33333334

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> opa = tfr.keras.metrics.OPAMetric(ragged=True)
  >>> opa(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.OPAMetric()])
  ```

  Definition:

  $$
  \text{OPA}(\{y\}, \{s\}) =
  \frac{\sum_i \sum_j I[s_i > s_j] I[y_i > y_j]}{\sum_i \sum_j I[y_i > y_j]}
  $$

  where $I[]$ is the indicator function:

  $$
  I[\text{cond}] = \begin{cases}
  1 & \text{if cond is true}\\
  0 & \text{else}\end{cases}
  $$
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(OPAMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._metric = metrics_impl.OPAMetric(name=name, ragged=ragged)
