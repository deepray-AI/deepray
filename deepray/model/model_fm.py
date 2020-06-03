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

from deepray.base.layers.interactions import FMNet
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS
flags.DEFINE_integer("fm_order", 2, "FM net polynomial order")
flags.DEFINE_integer("fm_rank", 2,
                     "Number of factors in low-rank appoximation.")
flags.DEFINE_integer("latent_factors", 10,
                     "Size of factors in low-rank appoximation.")


class FactorizationMachine(BaseCTRModel):

    def __init__(self, flags):
        super().__init__(flags)
        self.k = tf.constant(self.flags.latent_factors)

    def build(self, input_shape):
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        v = self.fm_block(features)
        return v

    def build_fm(self):
        return FMNet(k=self.k)
