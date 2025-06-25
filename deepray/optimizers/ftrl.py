import sys

import tensorflow as tf
from absl import flags

from deepray.custom_ops.embedding_variable import gen_kv_variable_ops
from deepray.custom_ops.embedding_variable import kv_variable_ops
from .ev_optimizer_patch import add_slot, SlotConfig


class FtrlOptimizer(tf.keras.optimizers.legacy.Ftrl):
  def __init__(self, learning_rate=0.001, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    self.global_step = None
    flags.FLAGS([sys.argv[0], f"--ev_slot_num={2}"])

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = tf.compat.v1.constant_initializer(self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, "accumulator", init, slot_config=SlotConfig(slot_index=1, slot_num=2))
      self.add_slot(var, "linear", slot_config=SlotConfig(slot_index=2, slot_num=2))

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)

    # Adjust L2 regularization strength to include beta to avoid the
    # underlying TensorFlow ops needing to include it.
    adjusted_l2_regularization_strength = coefficients["l2_regularization_strength"] + coefficients["beta"] / (
      2.0 * coefficients["lr_t"]
    )

    accum = self.get_slot(var, "accumulator")
    linear = self.get_slot(var, "linear")

    if self._l2_shrinkage_regularization_strength <= 0.0:
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        return gen_kv_variable_ops.kv_resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients["lr_t"],
          coefficients["l1_regularization_strength"],
          adjusted_l2_regularization_strength,
          coefficients["learning_rate_power"],
          use_locking=self._use_locking,
        )
      else:
        return tf.raw_ops.ResourceSparseApplyFtrl(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          indices=indices,
          lr=coefficients["lr_t"],
          l1=coefficients["l1_regularization_strength"],
          l2=adjusted_l2_regularization_strength,
          lr_power=coefficients["learning_rate_power"],
          use_locking=self._use_locking,
        )
    else:
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        return gen_kv_variable_ops.kv_resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients["lr_t"],
          coefficients["l1_regularization_strength"],
          adjusted_l2_regularization_strength,
          coefficients["l2_shrinkage_regularization_strength"],
          coefficients["learning_rate_power"],
          use_locking=self._use_locking,
        )
      else:
        return tf.raw_ops.ResourceSparseApplyFtrlV2(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          indices=indices,
          lr=coefficients["lr_t"],
          l1=coefficients["l1_regularization_strength"],
          l2=adjusted_l2_regularization_strength,
          l2_shrinkage=coefficients["l2_shrinkage_regularization_strength"],
          lr_power=coefficients["learning_rate_power"],
          use_locking=self._use_locking,
        )


FtrlOptimizer.add_slot = add_slot
