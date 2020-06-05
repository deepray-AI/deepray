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

from deepray.base.layers.interactions import CrossNet
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("cross_layers", 3, "number of cross layers")
flags.DEFINE_bool("cross_bias", False, "use_bias in cross")
flags.DEFINE_bool("sparse_cross", False, "sparse weights for cross")


class DeepCrossModel(BaseCTRModel):
    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.cross_block = self.build_cross(num_layers=self.flags.cross_layers)

    def build_network(self, features, is_training=None):
        """
        Deep & cross model

            cross: BN(log(dense)) + embedding

            deep: BN(log(dense)) + embedding
        """
        deep_out = self.deep_block(features, is_training=is_training)
        cross_out = self.cross_block(features, is_training=is_training)
        v = self.concat([deep_out, cross_out])
        return v

    def build_cross(self, num_layers=3):
        return CrossNet(num_layers,
                        use_bias=self.flags.cross_bias,
                        sparse=self.flags.sparse_cross, flags=self.flags)
