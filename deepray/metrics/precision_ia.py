from ._ranking import _RankingMetric


class PrecisionIAMetric(_RankingMetric):
  r"""Precision-IA@k (Pre-IA@k).

  Intent-aware Precision@k ([Agrawal et al, 2009][agrawal2009];
  [Clarke et al, 2009][clarke2009]) is a precision metric that operates on
  subtopics and is typically used for diversification tasks..

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  Pre-IA@k(y, s) = sum_t sum_i I[rank(s_i) <= k] y_{i,t} / (# of subtopics * k)
  ```

  NOTE: The labels `y_true` should be of shape
  `[batch_size, list_size, subtopic_size]`, indicating relevance for each
  subtopic in the last dimension.

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_{i,t} = 1` if `y_{i,t} >= 1`.

  Standalone usage:

  >>> y_true = [[[0., 1.], [1., 0.], [1., 1.]]]
  >>> y_pred = [[3., 1., 2.]]
  >>> pre_ia = tfr.keras.metrics.PrecisionIAMetric()
  >>> pre_ia(y_true, y_pred).numpy()
  0.6666667

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant(
  ...   [[[0., 0.], [1., 0.]], [[1., 1.], [0., 2.], [1., 0.]]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> pre_ia = tfr.keras.metrics.PrecisionIAMetric(ragged=True)
  >>> pre_ia(y_true, y_pred).numpy()
  0.5833334

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                metrics=[tfr.keras.metrics.PrecisionIAMetric()])
  ```

  Definition:

  $$
  \text{Pre-IA@k}(y, s) = \frac{1}{\text{# of subtopics} \cdot k}
  \sum_t \sum_i I[\text{rank}(s_i) \leq k] y_{i,t}
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly.

  References:

    - [Diversifying Search Results, Agrawal et al, 2009][agrawal2009]
    - [Overview of the TREC 2009 Web Track, Clarke et al, 2009][clarke2009]

  [agrawal2009]:
  https://www.microsoft.com/en-us/research/publication/diversifying-search-results/
  [clarke2009]: https://trec.nist.gov/pubs/trec18/papers/ENT09.OVERVIEW.pdf
  """

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    """Constructor.

    Args:
      name: A string used as the name for this metric.
      topn: A cutoff for how many examples to consider for this metric.
      dtype: Data type of the metric output. See `tf.keras.metrics.Metric`.
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True y_true, y_pred and sample_weight (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
      **kwargs: Other keyward arguments used in `tf.keras.metrics.Metric`.
    """
    super(PrecisionIAMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.PrecisionIAMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    config = super(PrecisionIAMetric, self).get_config()
    config.update({
      "topn": self._topn,
    })
    return config
