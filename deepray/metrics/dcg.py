from ._ranking import _RankingMetric


class DCGMetric(_RankingMetric):
  r"""Discounted cumulative gain (DCG).

  Discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002]).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```

  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> dcg = tfr.keras.metrics.DCGMetric()
  >>> dcg(y_true, y_pred).numpy()
  1.1309297

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> dcg = tfr.keras.metrics.DCGMetric(ragged=True)
  >>> dcg(y_true, y_pred).numpy()
  2.065465

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.DCGMetric()])
  ```

  Definition:

  $$
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
    super(DCGMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn or utils.pow_minus_1
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._metric = metrics_impl.DCGMetric(
      name=name, topn=topn, gain_fn=self._gain_fn, rank_discount_fn=self._rank_discount_fn, ragged=ragged
    )

  def get_config(self):
    base_config = super(DCGMetric, self).get_config()
    config = {
      "topn": self._topn,
      "gain_fn": self._gain_fn,
      "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config
