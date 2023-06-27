from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class PowerSign(optimizer.Optimizer):
  """Implementation of PowerSign.
  See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
  @@__init__
  """

  def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
    super(PowerSign, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._alpha = alpha
    self._beta = beta

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._alpha_t = None
    self._beta_t = None

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
    self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
    beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

    eps = 1e-7  # cap for moving average

    m = self.get_slot(var, "m")
    m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

    var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(tf.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value
    # Create an op that groups multiple operations.
    # When this op finishes, all ops in input have finished
    return control_flow_ops.group(*[var_update, m_t])

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
