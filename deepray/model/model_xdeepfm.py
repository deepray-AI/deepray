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
Author:
    Hailin Fu, hailinfufu@outlook.com
"""
from absl import flags

from deepray.base.layers.cin import CompressedInteractionNetwork

from deepray.base.layers.core import Linear
from deepray.model.model_flen import FLENModel
from deepray.model.model_fm import FactorizationMachine
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean("res_cin",
                     False,
                     "Whether use residual structure to fuse the results from each layer of CIN.")
flags.DEFINE_boolean("split_half", False, "")
flags.DEFINE_string("cin_layers", "128,128",
                    "sizes of CIN layers, string delimited by comma")


class ExtremeDeepFMModel(FactorizationMachine, FLENModel):
    def __init__(self, flags):
        super(ExtremeDeepFMModel, self).__init__(flags)
        self.flags = flags

    def build(self, input_shape):
        dnn_hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        cin_hidden = [int(h) for h in self.flags.cin_layers.split(',')]
        self.embeddingsize = self.flags.embed_dim
        self.numbericfeaturescnt = len(self.NUMERICAL_FEATURES)
        self.fm_block = self.build_fm()
        self.linear_block = Linear()
        self.dnn_block = self.build_deep(dnn_hidden)
        self.cin_block = self.build_cin(cin_hidden)

    def build_network(self, features, is_training=None):
        ev_list, fv_list = features
        ev_list = self.concat(ev_list)
        fv_list = tf.reshape(
            tf.reshape(fv_list, [-1, self.numbericfeaturescnt, 1]) * tf.ones([self.embeddingsize], tf.float32),
            [-1, self.embeddingsize * self.numbericfeaturescnt])
        l = self.concat([ev_list, fv_list])
        x_cin = tf.reshape(l, [-1, l.shape[1] // self.embeddingsize, self.embeddingsize])
        cin_part = self.cin_block(x_cin)
        dnn_part = self.dnn_block(l)
        fm_part = self.fm_block(l)
        linear_part = self.linear_block(l)
        logit = self.concat([linear_part, fm_part, dnn_part, cin_part])
        return logit

    def build_cin(self, hidden):
        return CompressedInteractionNetwork(layer_size=hidden,
                                            activation='relu',
                                            split_half=self.flags.split_half,
                                            use_res=self.flags.res_cin)
