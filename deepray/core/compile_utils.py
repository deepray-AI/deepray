import horovod.tensorflow as hvd
import tensorflow as tf
from absl import flags
from packaging.version import parse

if parse(tf.__version__.replace("-tf", "+tf")) < parse("2.11"):
  from keras.engine.compile_utils import MetricsContainer, match_dtype_and_rank, get_mask, apply_mask
elif parse(tf.__version__) > parse("2.16.0"):
  from tf_keras.src.engine.compile_utils import MetricsContainer, match_dtype_and_rank
  from tf_keras.src.utils.losses_utils import get_mask, apply_mask
else:
  from keras.src.engine.compile_utils import MetricsContainer, match_dtype_and_rank
  from keras.src.utils.losses_utils import get_mask, apply_mask


class HvdMetricsContainer(MetricsContainer):
  def update_state(self, y_true, y_pred, sample_weight=None):
    """Updates the state of per-output metrics."""
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)

    if not self._built:
      self.build(y_pred, y_true)

    y_pred = tf.nest.flatten(y_pred)
    y_true = tf.nest.flatten(y_true) if y_true is not None else []
    sample_weight = tf.nest.flatten(sample_weight)

    zip_args = (
      y_true,
      y_pred,
      sample_weight,
      self._metrics,
      self._weighted_metrics,
    )
    for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
      # Ok to have no metrics for an output.
      if y_t is None or (all(m is None for m in metric_objs) and all(wm is None for wm in weighted_metric_objs)):
        continue

      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      mask = get_mask(y_p)
      sw = apply_mask(y_p, sw, mask)

      if flags.FLAGS.use_horovod:
        y_t = hvd.allgather(y_t)
        y_p = hvd.allgather(y_p)
        if mask is not None:
          mask = hvd.allgather(mask)
        if sw is not None:
          sw = hvd.allgather(sw)

      for metric_obj in metric_objs:
        if metric_obj is None:
          continue
        metric_obj.update_state(y_t, y_p, sample_weight=mask)

      for weighted_metric_obj in weighted_metric_objs:
        if weighted_metric_obj is None:
          continue
        weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)
