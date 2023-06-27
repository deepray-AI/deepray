import tensorflow as tf

from deepray.io.datasets.movielens import constants as rconst


class HitRateMetric(tf.keras.metrics.Metric):
  def __init__(self, match_mlperf, name='hit_rate', **kwargs):
    super(HitRateMetric, self).__init__(name=name, **kwargs)
    self.hr_sum = self.add_weight(name='hr_sum', initializer='zeros')
    self.hr_count = self.add_weight(name='hr_count', initializer='zeros')
    self._hit_rate = self.add_weight(name='hit_rate', initializer='zeros')
    self._match_mlperf = match_mlperf

  def update_state(self, logits, dup_mask):
    dup_mask = tf.cast(dup_mask, tf.float32)
    logits = tf.slice(logits, [0, 1], [-1, -1])
    in_top_k, _, metric_weights, _ = self.compute_top_k_and_ndcg(
      logits, dup_mask, self._match_mlperf)
    metric_weights = tf.cast(metric_weights, tf.float32)

    hr_sum = tf.reduce_sum(in_top_k * metric_weights)
    hr_count = tf.reduce_sum(metric_weights)
    self.hr_sum.assign_add(hr_sum)
    self.hr_count.assign_add(hr_count)

  def compute_top_k_and_ndcg(self,
                             logits: tf.Tensor,
                             duplicate_mask: tf.Tensor,
                             match_mlperf: bool = False):
    """Compute inputs of metric calculation.

    Args:
      logits: A tensor containing the predicted logits for each user. The shape of
        logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits for a
        user are grouped, and the first element of the group is the true element.
      duplicate_mask: A vector with the same shape as logits, with a value of 1 if
        the item corresponding to the logit at that position has already appeared
        for that user.
      match_mlperf: Use the MLPerf reference convention for computing rank.

    Returns:
      is_top_k, ndcg and weights, all of which has size (num_users_in_batch,), and
      logits_by_user which has size
      (num_users_in_batch, (rconst.NUM_EVAL_NEGATIVES + 1)).
    """
    logits_by_user = tf.reshape(logits, (-1, rconst.NUM_EVAL_NEGATIVES + 1))
    duplicate_mask_by_user = tf.cast(
      tf.reshape(duplicate_mask, (-1, rconst.NUM_EVAL_NEGATIVES + 1)),
      logits_by_user.dtype)

    if match_mlperf:
      # Set duplicate logits to the min value for that dtype. The MLPerf
      # reference dedupes during evaluation.
      logits_by_user *= (1 - duplicate_mask_by_user)
      logits_by_user += duplicate_mask_by_user * logits_by_user.dtype.min

    # Determine the location of the first element in each row after the elements
    # are sorted.
    sort_indices = tf.argsort(logits_by_user, axis=1, direction="DESCENDING")

    # Use matrix multiplication to extract the position of the true item from the
    # tensor of sorted indices. This approach is chosen because both GPUs and TPUs
    # perform matrix multiplications very quickly. This is similar to np.argwhere.
    # However this is a special case because the target will only appear in
    # sort_indices once.
    one_hot_position = tf.cast(
      tf.equal(sort_indices, rconst.NUM_EVAL_NEGATIVES), tf.int32)
    sparse_positions = tf.multiply(
      one_hot_position,
      tf.range(logits_by_user.shape[1])[tf.newaxis, :])
    position_vector = tf.reduce_sum(sparse_positions, axis=1)

    in_top_k = tf.cast(tf.less(position_vector, rconst.TOP_K), tf.float32)
    ndcg = tf.math.log(2.) / tf.math.log(tf.cast(position_vector, tf.float32) + 2)
    ndcg *= in_top_k

    # If a row is a padded row, all but the first element will be a duplicate.
    metric_weights = tf.not_equal(
      tf.reduce_sum(duplicate_mask_by_user, axis=1), rconst.NUM_EVAL_NEGATIVES)

    return in_top_k, ndcg, metric_weights, logits_by_user

  def result(self):
    return self.hr_sum / self.hr_count

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.hr_sum.assign(0.)
    self.hr_count.assign(0.)
    self._hit_rate.assign(0.)

  def get_config(self):
    return {"match_mlperf": self._match_mlperf}

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

