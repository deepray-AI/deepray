import tensorflow as tf
from tensorflow.keras.initializers import (Zeros, glorot_normal)
from tensorflow.python.keras.regularizers import l2


class CrossNet(tf.keras.layers.Layer):
  def __init__(self, layer_num=2, l2_reg=0, seed=1024, **kwargs):
    super(CrossNet, self).__init__(**kwargs)
    self.layer_num = layer_num
    self.l2_reg = l2_reg
    self.seed = seed

  def build(self, input_shape):
    # shape = tf.TensorShape((input_shape[1], input_shape[1]))
    dim = input_shape[-1]
    self.kernels = [self.add_weight(name='kernal' + str(i),
                                    shape=(dim, 1),
                                    initializer=glorot_normal(seed=self.seed),
                                    regularizer=l2(self.l2_reg),
                                    trainable=True) for i in range(self.layer_num)]
    self.bias = [self.add_weight(name='bias' + str(i),
                                 shape=(dim, 1),
                                 initializer=Zeros(),
                                 trainable=True) for i in range(self.layer_num)]
    super(CrossNet, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x_0 = tf.expand_dims(inputs, axis=2)
    x_l = x_0
    for i in range(self.layer_num):
      xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
      dot_ = tf.matmul(x_0, xl_w)
      x_l = dot_ + self.bias[i] + x_l
    x_l = tf.squeeze(x_l, axis=2)
    return x_l

  def get_config(self):
    config = {'layer_num': self.layer_num,
              'l2_reg': self.l2_reg,
              'seed': self.seed
              }
    base_config = super(CrossNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = input_shape[1]
    return tf.TensorShape(shape)
