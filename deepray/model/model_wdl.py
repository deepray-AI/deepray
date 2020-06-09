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

from deepray.base.layers.core import Linear
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS

flags.DEFINE_list('wide_cols', None, 'Features List for wide part')
flags.DEFINE_list('deep_cols', None, 'Features List for deep part')


class WideAndDeepModel(BaseCTRModel):
    voc_emb_size = None

    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.wide_block = self.build_wide(hidden=1)

    def build_network(self, features, is_training=None):
        wide_part, deep_part = features[0], features[1]
        deep_part = self.deep_block(deep_part, is_training=is_training)
        wide_part = self.wide_block(wide_part, is_training=is_training)
        logit = tf.concat([wide_part, deep_part], -1)
        return logit

    def build_features(self, features, embedding_suffix=''):
        """
        categorical feature id starts from -1 (as missing)
        """
        if self.flags.wide_cols:
            ev_list = [self.EmbeddingDict[key](features[key])
                       for key in self.CATEGORY_FEATURES if key in self.flags.wide_cols]
            fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES if
                       key in self.flags.wide_cols]
            wide_list = self.concat(fv_list + ev_list)
        else:
            ev_list = [self.EmbeddingDict[key](features[key])
                       for key in self.CATEGORY_FEATURES]
            fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
            wide_list = self.concat(fv_list + ev_list)
        if self.flags.deep_cols:
            ev_list = [self.EmbeddingDict[key](features[key])
                       for key in self.CATEGORY_FEATURES if key in self.flags.deep_cols]
            fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES if
                       key in self.flags.deep_cols]
            deep_list = self.concat(fv_list + ev_list)
        else:
            ev_list = [self.EmbeddingDict[key](features[key])
                       for key in self.CATEGORY_FEATURES]
            fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
            deep_list = self.concat(fv_list + ev_list)

        return tf.concat(wide_list, -1), tf.concat(deep_list, -1)

    def build_wide(self, hidden=1):
        return Linear(hidden)
