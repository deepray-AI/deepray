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

from deepray.model.model_fm import FactorizationMachine


class NeuralFactorizationMachine(FactorizationMachine):
    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.B_Interaction_Layer = self.build_fm()
        self.Hidden_Layers = self.build_deep(hidden)

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        v = self.B_Interaction_Layer(features)
        v = self.Hidden_Layers(v)
        return v
