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
