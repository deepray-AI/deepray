#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#       Hailin  <hailinfufu@outlook>
#  Distributed under terms of the GNU license.
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
