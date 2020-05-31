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

from deepray.model.model_fm import FactorizationMachine


class AttentionalFactorizationMachine(FactorizationMachine):
    def build(self, input_shape):
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        v = self.fm_block(input)
        return v
