#  -*- coding: utf-8 -*-
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================

"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""

from deepray.model.model_ctr import BaseCTRModel


class FMNeuralNetwork(BaseCTRModel):
    def __init__(self):
        super(FMNeuralNetwork, self).__init__()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
