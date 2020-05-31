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
from absl import flags

from deepray.base.layers.interactions import SelfAttentionNet
from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("heads", 2, "number of heads")
flags.DEFINE_integer("field_size", 23, "number of fields")
flags.DEFINE_integer("blocks", 2, "number of blocks")
flags.DEFINE_string("block_shape", "16,16",
                    "output shape of each block")
flags.DEFINE_bool("has_residual", False, "add has_residual")


class AutoIntModel(BaseCTRModel):

    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.attention_block = self.build_attention(concat_last_deep=True)

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        deep_part = self.deep_block(features, is_training=is_training)
        attention_part = self.attention_block(features, is_training=is_training)
        v = tf.concat([attention_part, deep_part], -1)
        return v

    def build_attention(self, concat_last_deep):
        hidden = [16, 16]  # [int(h) for h in self.flags.deep_layers.split(',')]
        return SelfAttentionNet(hidden=hidden, concat_last_deep=concat_last_deep)
