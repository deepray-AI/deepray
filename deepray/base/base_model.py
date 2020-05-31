#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================

"""
Base model, just build model, not working without model_train_base.

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import tensorflow as tf
from absl import flags

from deepray.base.layers.core import DeepNet

FLAGS = flags.FLAGS

flags.DEFINE_bool("use_bn", False, "Whether use BatchNormalization in Dense layer")
flags.DEFINE_bool("res_deep", False, "residual connections for deep")
flags.DEFINE_bool("sparse_deep", False, "sparse weights for deep")
flags.DEFINE_bool("concat_last_deep", False,
                  "use the deep output with the self attentive output")
flags.DEFINE_bool("renorm", False, "renorm in BN")

flags.DEFINE_enum("summary_mode", "loss", ["loss", "all"],
                  "tf.summary, loss: loss scalar only; "
                  "all: summary for all scalar and histogram.")
flags.DEFINE_float("l1", 0.1, "l1 regularization")
flags.DEFINE_float("l2", 0.1, "l2 regularization")
flags.DEFINE_float("decoupled_weight_decay", 0,
                   "if > 0, using extend_with_decoupled_weight_decay")
flags.DEFINE_float("keep_prob", 1.0, "keep prob for dropout")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_float("ns_rate", 1, 'negative sampling rate, keep prob')
flags.DEFINE_integer("bucket_peak", 9, "tanh(9)=0.99999997")
flags.DEFINE_integer("embedding_size", 50, "embedding size")
flags.DEFINE_integer("emb_size_factor", 6, "emb size is "
                                           "emb_size_factor * (voc_size ** 0.25)")
flags.DEFINE_integer("factor_size", 64, "factor size for mvm")
flags.DEFINE_integer("hash_bits", 20,
                     "hash bits for categorical features")
flags.DEFINE_integer("inference_num_position", 10,
                     "number of positions for inference")
flags.DEFINE_integer("lr_decay_step", 0,
                     "learning rate first decay step "
                     "for cosine_decay_restarts")
flags.DEFINE_float("lr_schedule_max", 10.0,
                   "upper bound for learning rate update")
flags.DEFINE_integer("lr_schedule_update_step", 1,
                     "step is used for smoothing lr scheduling, set to 1 to update lr per every minibatch")
flags.DEFINE_integer("lr_schedule_print_step", 100,
                     "print out every N step")
flags.DEFINE_integer("num_buckets", 100,
                     "number of buckets on continuous features")
flags.DEFINE_string("deep_layers", "1024,1024",
                    "sizes of deep layers, string delimited by comma")
flags.DEFINE_string("dense_minmax_dir", "", "dense minmax summary file")
flags.DEFINE_string("img_npz", "", "npz for images")
flags.DEFINE_string("iv_npz", "", "npz for image vector")
flags.DEFINE_string("model", "lr", "model")
flags.DEFINE_string("voc_dir", "", "vocabulary size summary dir")

PREDICT_LAYER_NAME = "predict"


class BaseModel(tf.keras.models.Model):
    def __init__(self, flags):
        super().__init__(flags)
        # build by subclass' build function
        self.costs = None
        self.global_step = None
        self.flags = flags

    def BN(self, fv):
        with tf.name_scope('BN'):
            fv = tf.keras.layers.BatchNormalization(fused=True,
                                                    renorm=self.flags.renorm)(fv, training=self.is_training, )
            return fv

    def concat(self, inputs):
        return tf.concat(inputs, -1)

    def build_deep(self, hidden=None, activation=tf.nn.relu):
        return DeepNet(hidden, activation, sparse=False, droprate=1 - self.flags.keep_prob, flags=self.flags)

    def build_dense_layer(self, fv):
        if self.flags.use_bn:
            return self.BN(tf.math.log1p(fv))
        else:
            return tf.math.log1p(fv)

    def build_predictions(self):
        if self.VOC_SIZE[self.LABEL] == 2:
            prediction = tf.keras.layers.Dense(
                1, activation=tf.nn.sigmoid, name=PREDICT_LAYER_NAME + "/Sigmoid")
        else:
            prediction = tf.keras.layers.Dense(
                self.VOC_SIZE[self.LABEL],
                activation=tf.nn.softmax,
                name=PREDICT_LAYER_NAME + "/Softmax")
        return prediction
