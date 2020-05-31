#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================

"""
trainer

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import time

import tensorflow as tf
from absl import logging


def train(model):
    logging.info("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    # strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    # with strategy.scope():
    start_time = time.time()
    history = model.train(model)
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    return history
