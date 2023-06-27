#!/usr/bin/env python
# @Time    : 2021/8/12 8:18 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import tensorflow as tf


class FactorizationMachine(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 2D tensor with shape: ``(batch_size, embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
        - [DeepRec](https://github.com/alibaba/DeepRec/blob/main/modelzoo/deepfm/train.py)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        # Calculate output with FM equation
        sum_square = tf.square(tf.reduce_sum(inputs, axis=1))
        square_sum = tf.reduce_sum(tf.square(inputs), axis=1)
        outputs = 0.5 * tf.subtract(sum_square, square_sum)
        return outputs
