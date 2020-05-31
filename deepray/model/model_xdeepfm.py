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

from absl import flags

from deepray.base.layers.compressed_interaction_network import CIN
from deepray.base.layers.core import Linear
from deepray.model.model_fm import FactorizationMachine
from deepray.model.model_flen import FLENModel

FLAGS = flags.FLAGS
flags.DEFINE_boolean("res_cin",
                     False,
                     "Whether use residual structure to fuse the results from each layer of CIN.")
flags.DEFINE_boolean("reduce_D", False, "")
flags.DEFINE_boolean("cin_bias", False, "Whether to add bias term in CIN.")
flags.DEFINE_boolean("cin_direct",
                     False,
                     "If true, then all hidden units are connected to both next layer and output layer;"
                     "otherwise, half of hidden units are connected to next layer and the other half will be "
                     "connected to output layer.")
flags.DEFINE_string("cin_layers", "128,128",
                    "sizes of CIN layers, string delimited by comma")


class ExtremeDeepFMModel(FactorizationMachine, FLENModel):
    def __init__(self, flags):
        super(ExtremeDeepFMModel, self).__init__(flags)
        self.flags = flags

    def build(self, input_shape):
        dnn_hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        cin_hidden = [int(h) for h in self.flags.cin_layers.split(',')]
        self.fm_block = self.build_fm()
        self.linear_block = Linear()
        self.dnn_block = self.build_deep(dnn_hidden)
        self.cin_block = self.build_cin(cin_hidden)

    def build_network(self, features, is_training=None):
        ev_list, fv_list = features
        cin_part = self.cin_block(ev_list)
        dnn_part = self.dnn_block(self.concat(ev_list + fv_list))
        fm_part = self.fm_block(self.concat(ev_list + fv_list))
        linear_part = self.linear_block(self.concat(fv_list + fv_list))
        v = self.concat([linear_part, fm_part, dnn_part, cin_part])
        return v

    def build_cin(self, hidden):
        return CIN(hidden=hidden,
                   activation='relu',
                   use_bias=self.flags.cin_bias,
                   use_direct=self.flags.cin_direct,
                   use_reduce_D=self.flags.reduce_D,
                   use_res=self.flags.res_cin)