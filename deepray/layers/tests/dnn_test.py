import tensorflow as tf
from typing import List
from ..dnn import DNN


def test_DNN():
  # 测试DNN类的初始化
  dnn = DNN(dnn_hidden_units=[8, 16], use_bn=True)
  assert len(dnn.kernel) == 2
  assert isinstance(dnn.bn, tf.keras.layers.BatchNormalization)
  assert dnn._fn == dnn.apply_kernel_bn

  # 测试DNN类的call方法
  x = tf.ones((4, 4))
  y = dnn(x)
  assert y.shape == (4, 16)

  # 测试DNN类的apply_kernel方法
  x = tf.ones((4, 4))
  y = dnn.apply_kernel(0, x)
  assert y.shape == (4, 8)

  # 测试DNN类的apply_kernel_bn方法
  x = tf.ones((4, 4))
  y = dnn.apply_kernel_bn(0, x)
  assert y.shape == (4, 8)
