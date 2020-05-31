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


class ProductNeuralNetwork(BaseCTRModel):
    def __init__(self):
        super(ProductNeuralNetwork, self).__init__()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
