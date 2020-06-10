#  Copyright © 2020-2020 Hailin Fu All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
"""
Base model, just build model, not working without model_train_base.

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import tensorflow as tf
from absl import flags

from deepray.base.layers.core import DeepNet


flags.DEFINE_bool("use_bn", False, "Whether use BatchNormalization in Dense layer")
flags.DEFINE_bool("res_deep", False, "residual connections for deep")
flags.DEFINE_bool("renorm", False, "renorm in BN")
flags.DEFINE_float("l1", 0.1, "l1 regularization")
flags.DEFINE_float("l2", 0.1, "l2 regularization")
flags.DEFINE_float("keep_prob", 1.0, "keep prob for dropout")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("embedding_size", 50, "embedding size")
flags.DEFINE_integer("emb_size_factor", 6, "emb size is "
                                           "emb_size_factor * (voc_size ** 0.25)")
flags.DEFINE_string("deep_layers", "1024,1024",
                    "sizes of deep layers, string delimited by comma")
PREDICT_LAYER_NAME = "prediction"


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
