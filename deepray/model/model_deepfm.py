#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================


"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""

import tensorflow as tf

from deepray.model.model_fm import FactorizationMachine


class DeepFM(FactorizationMachine):

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        fm_out = self.fm_block(features)
        deep_out = self.deep_block(features)
        v = tf.concat([deep_out, fm_out], -1)
        return v
