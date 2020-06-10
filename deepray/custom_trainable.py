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
Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import csv
import os

import tensorflow as tf
from absl import flags

from deepray.base.base_trainable import BaseTrainable

FLAGS = flags.FLAGS
flags.DEFINE_string("feature_map", None, "path to feature_map")
flags.DEFINE_string("black_list", None, "black list for feature_map")
flags.DEFINE_bool("skip_varLen_feature", False,
                  "skip using var length feature")


class CustomTrainable(BaseTrainable, object):

    def __init__(self, flags):
        super(CustomTrainable, self).__init__(flags)
        global LABEL, CATEGORY_FEATURES, NUMERICAL_FEATURES, VOC_SIZE, VARIABLE_FEATURES
        LABEL, CATEGORY_FEATURES, NUMERICAL_FEATURES, VOC_SIZE, VARIABLE_FEATURES = self.LABEL, self.CATEGORY_FEATURES, self.NUMERICAL_FEATURES, self.VOC_SIZE, self.VARIABLE_FEATURES

    @classmethod
    def parser(cls, record):
        flags = FLAGS
        feature_map = {}
        if not flags.skip_varLen_feature:
            for key, dim in VARIABLE_FEATURES.items():
                feature_map[key] = tf.io.VarLenFeature(tf.int64)
        for key, dim in CATEGORY_FEATURES.items():
            feature_map[key] = tf.io.FixedLenFeature([1], tf.int64)
        for key, dim in NUMERICAL_FEATURES.items():
            feature_map[key] = tf.io.FixedLenFeature([1], tf.float32)
        features = tf.io.parse_single_example(record, features=feature_map)
        label = tf.io.parse_single_example(record,
                                           features={LABEL: tf.io.FixedLenFeature([1], tf.int64)})
        return features, label[LABEL]

    @classmethod
    def get_summary(cls):
        black_list = flags.FLAGS.black_list
        with open(os.path.join(os.path.dirname(__file__),
                               FLAGS.feature_map)) as f:
            feature_list = list(csv.reader(f))
        if black_list:
            with open(black_list) as f:
                black_feature_list = [feature.strip() for feature in f]
            feature_list = list(filter(lambda x: x[1] not in black_feature_list, feature_list))
        label = [item[1] for item in feature_list if item[2] == 'LABEL']
        category_features = {item[1]: int(item[0]) for item in feature_list if item[2] == 'CATEGORICAL'}
        numerical_features = {item[1]: int(item[0]) for item in feature_list if item[2] == 'NUMERICAL'}
        variable_features = {item[1]: int(item[0]) for item in feature_list if item[2] == 'VARIABLE'}
        voc_size = {item[1]: int(item[0]) for item in feature_list if item[2] in ['CATEGORICAL', 'VARIABLE', 'LABEL']}
        return label[0], category_features, numerical_features, voc_size, variable_features
