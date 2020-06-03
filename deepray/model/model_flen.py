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

from deepray.base.layers.core import CustomDropout, DeepBlock
from deepray.base.layers.field_wise_bi_interaction import FieldWiseBiInteraction
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("embed_dim", 4, "Embedding dim for all categorical feature, cause FLEN doesn't support "
                                     "embedding dim automatic inference")


class FLENModel(BaseCTRModel):

    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.mlp_block = self.build_deep(hidden=hidden)

        # field-weighted embedding
        self.fwbi = FieldWiseBiInteraction()
        self.fwbi_fc_32 = DeepBlock(hidden=32, activation=tf.nn.relu, prefix='fwbi_fc', sparse=None,
                                    flags=self.flags)
        self.fwbi_bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.fwbi_drop = CustomDropout(rate=0.2)

        # self.fwbi_block = self.build_fwbi()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        ev_list, fv_list = features
        deep_out = self.mlp_block(self.concat(ev_list + fv_list))
        # field-weighted embedding
        fwbi = self.fwbi(ev_list)
        fwbi_fc_32 = self.fwbi_fc_32(fwbi)
        fwbi_bn = self.fwbi_bn(fwbi_fc_32)
        fwbi_out = self.fwbi_drop(fwbi_bn)
        # fwbi_out = self.fwbi_block(features)
        v = tf.concat(values=[deep_out, fwbi_out], axis=1)
        return v

    def build_features(self, features, embedding_suffix=''):
        """
        categorical feature id starts from -1 (as missing)
        """
        ev_list = [self.EmbeddingDict[key](features[key])
                   for key in self.CATEGORY_FEATURES]
        fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
        return ev_list, fv_list

    def load_voc_summary(self):
        """
        model doesn't support categorical feature custom embedding size
        """
        voc_emb_size = dict()
        for key, voc_size in self.VOC_SIZE.items():
            emb_size = self.flags.embed_dim
            voc_emb_size[key] = [voc_size, emb_size]
        for k, v in voc_emb_size.items():
            if k == self.LABEL:
                continue
            self.print_emb_info(k, v[0], v[1])
        return voc_emb_size
