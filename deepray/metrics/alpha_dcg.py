from ._ranking import _RankingMetric


class AlphaDCGMetric(_RankingMetric):
  r"""Alpha discounted cumulative gain (alphaDCG).

  Alpha discounted cumulative gain ([Clarke et al, 2008][clarke2008];
  [Clarke et al, 2009][clarke2009]) is a cumulative gain metric that operates
  on subtopics and is typically used for diversification tasks.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  alphaDCG(y, s) = sum_t sum_i gain(y_{i,t}) * rank_discount(rank(s_i))
  gain(y_{i,t}) = (1 - alpha)^(sum_j I[rank(s_j) < rank(s_i)] * gain(y_{j,t}))
  ```

  NOTE: The labels `y_true` should be of shape
  `[batch_size, list_size, subtopic_size]`, indicating relevance for each
  subtopic in the last dimension.

  NOTE: The `rank_discount_fn` should be keras serializable. Please see
  `tfr.keras.utils.log2_inverse` as an example when defining user customized
  functions.

  Standalone usage:

  >>> y_true = [[[0., 1.], [1., 0.], [1., 1.]]]
  >>> y_pred = [[3., 1., 2.]]
  >>> alpha_dcg = tfr.keras.metrics.AlphaDCGMetric()
  >>> alpha_dcg(y_true, y_pred).numpy()
  2.1963947

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant(
  ...   [[[0., 0.], [1., 0.]], [[1., 1.], [0., 2.], [1., 0.]]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> alpha_dcg = tfr.keras.metrics.AlphaDCGMetric(ragged=True)
  >>> alpha_dcg(y_true, y_pred).numpy()
  1.8184297

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.AlphaDCGMetric()])
  ```

  Definition:

  $$
  \alpha\text{DCG}(y, s) =
  \sum_t \sum_i \text{gain}(y_{i, t}, \alpha)
  \text{ rank_discount}(\text{rank}(s_i))\\
  \text{gain}(y_{i, t}, \alpha) =
  y_{i, t} (1 - \alpha)^{\sum_j I[\text{rank}(s_j) < \text{rank}(s_i)] y_{j, t}}
  $$

  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly and $I[]$ is the indicator function:

  $$
  I[\text{cond}] = \begin{cases}
  1 & \text{if cond is true}\\
  0 & \text{else}\end{cases}
  $$

  References:

    - [Novelty and diversity in information retrieval evaluation, Clarke et al,
       2008][clarke2008]
    - [Overview of the TREC 2009 Web Track, Clarke et al, 2009][clarke2009]

  [clarke2008]: https://dl.acm.org/doi/10.1145/1390334.1390446
  [clarke2009]: https://trec.nist.gov/pubs/trec18/papers/ENT09.OVERVIEW.pdf
  """

  def __init__(
    self,
    name="alpha_dcg_metric",
    topn=None,
    alpha=0.5,
    rank_discount_fn=None,
    seed=None,
    dtype=None,
    ragged=False,
    **kwargs,
  ):
    """Construct the ranking metric class for alpha-DCG.

    Args:
      name: A string used as the name for this metric.
      topn: A cutoff for how many examples to consider for this metric.
      alpha: A float between 0 and 1, parameter used in definition of alpha-DCG.
        Introduced as an assessor error in judging whether a document is
        covering a subtopic of the query.
      rank_discount_fn: A function of rank discounts. Default is set to
        `1 / log2(rank+1)`. The `rank_discount_fn` should be keras serializable.
        Please see the `log2_inverse` above as an example when defining user
        customized functions.
      seed: The ops-level random seed used in shuffle ties in `sort_by_scores`.
      dtype: Data type of the metric output. See `tf.keras.metrics.Metric`.
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True y_true, y_pred and sample_weight (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
      **kwargs: Other keyward arguments used in `tf.keras.metrics.Metric`.
    """
    super(AlphaDCGMetric, self).__init__(name=name, dtype=dtype, ragged=ragged, **kwargs)
    self._topn = topn
    self._alpha = alpha
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._seed = seed
    self._metric = metrics_impl.AlphaDCGMetric(
      name=name, topn=topn, alpha=alpha, rank_discount_fn=self._rank_discount_fn, seed=seed, ragged=ragged
    )

  def get_config(self):
    config = super(AlphaDCGMetric, self).get_config()
    config.update({
      "topn": self._topn,
      "alpha": self._alpha,
      "rank_discount_fn": self._rank_discount_fn,
      "seed": self._seed,
    })
    return config
