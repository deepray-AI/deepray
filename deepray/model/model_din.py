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

import tensorflow as tf
from absl import flags

from deepray.base.layers.attention import LocalActivationUnit
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS

flags.DEFINE_string("history_item", "hist_iid",
                    "feature name of user historical sequence")
flags.DEFINE_string("candidate_item", "hist_cate_id",
                    "feature name of candidate item sequence")


class DeepInterestNetwork(BaseCTRModel):

    def __init__(self, flags):
        super().__init__(flags)
        self.flags = flags

    def build(self, input_shape):
        # 1. mlp part
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.mlp_block = self.build_deep(hidden=hidden)

        # 2. DIN
        self.din_block = self.build_din()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
        ev_list, sparse_ev_list, fv_list = features

        din_part = self.din_block(sparse_ev_list[self.flags.candidate_item],
                                  sparse_ev_list[self.flags.history_item],
                                  is_training)
        deep_part = self.mlp_block(self.concat(ev_list + fv_list))
        v = tf.concat(values=[din_part, deep_part], axis=1)
        return v

    def build_features(self, features, embedding_suffix=''):
        """
        categorical feature id starts from -1 (as missing)
        """
        ev_list = [self.EmbeddingDict[key](features[key])
                   for key in self.CATEGORY_FEATURES]
        sparse_ev_list = {key: self.EmbeddingDict[key](features[key],
                                                       combiner=self.flags.sparse_embedding_combiner) for key in
                          self.VARLEN_FEATURES}
        fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
        return ev_list, sparse_ev_list, fv_list

    def build_din(self):
        return LocalActivationUnit()
