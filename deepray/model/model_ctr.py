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

Base Deepray model with base network and helper functions.
"""

import tensorflow as tf
from absl import flags, logging

from deepray.base.base_model import BaseModel
from deepray.base.layers.embedding import CustomEmbedding
from deepray.custom_trainable import CustomTrainable

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sparse_embedding_combiner", "mean", ["mean", "sqrtn"],
    "mean is the weighted sum divided by the total weight."
    "sqrtn is the weighted sum divided by the square root of the sum of "
    "the squares of the weights.")


class BaseCTRModel(CustomTrainable, BaseModel):
    def __init__(self, flags):
        super().__init__(flags)
        BaseCTRModel.voc_emb_size = self.load_voc_summary()
        self.EmbeddingDict = self.build_EmbeddingDict()
        self.predict_layer = self.build_predictions()

    def build_EmbeddingDict(self):
        return {
            feat: CustomEmbedding(feat,
                                  input_dim=self.voc_emb_size[feat][0],
                                  output_dim=self.voc_emb_size[feat][1],
                                  name_scope='embedding',
                                  initial_range=None) for feat in {**self.CATEGORY_FEATURES, **self.VARIABLE_FEATURES}}

    def build_features(self, features, embedding_suffix=''):
        """
        categorical feature id starts from -1 (as missing)
        """
        ev_list = [self.EmbeddingDict[key](features[key])
                   for key in self.CATEGORY_FEATURES]
        sparse_ev_list = []
        if not self.flags.skip_varLen_feature:
            for key in self.VARIABLE_FEATURES:
                sparse_ev_list.append(
                    self.EmbeddingDict[key](features[key],
                                            combiner=self.flags.sparse_embedding_combiner))

        fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
        if self.flags.summary_mode == 'all':
            for fv in fv_list:
                tf.summary.histogram(fv.name, fv)
        inputs = self.concat(fv_list + ev_list + sparse_ev_list)
        return inputs

    @classmethod
    def compute_emb_size(cls, voc_size):
        return int(FLAGS.emb_size_factor * (voc_size ** 0.25))

    @classmethod
    def print_emb_info(cls, feature_name, voc_size, emb_size):
        logging.info(" voc_size{:8} emb_size{:4} feature_name {}".format(
            voc_size, emb_size, feature_name))

    @classmethod
    def tfrecord_pipeline(cls, tfrecord_file, batch_size,
                          epochs, shuffle=True):
        return CustomTrainable.tfrecord_pipeline(tfrecord_file, batch_size,
                                                 epochs, shuffle)

    def reshape_input(self, features):
        reshaped = dict()
        for key, value in features.items():
            if key in self.CATEGORY_FEATURES or self.VARIABLE_FEATURES:
                reshaped[key] = value
            if key in self.NUMERICAL_FEATURES or key == self.LABEL:
                reshaped[key] = tf.reshape(
                    value,
                    [-1, 1])
        return reshaped

    def load_voc_summary(self):
        voc_emb_size = dict()
        for key, voc_size in self.VOC_SIZE.items():
            emb_size = self.compute_emb_size(voc_size)
            voc_emb_size[key] = [voc_size + 1, emb_size]
        for k, v in voc_emb_size.items():
            if k == self.LABEL:
                continue
            self.print_emb_info(k, v[0], v[1])
        return voc_emb_size

    def call(self, inputs, is_training=None, mask=None):
        features = self.build_features(inputs)
        logit = self.build_network(features, is_training)
        preds = self.predict_layer(logit)
        if self.flags.ns_rate < 1:
            preds = preds / (preds + tf.divide(tf.subtract(1.0, preds),
                                               self.flags.ns_rate))
        return preds

    def build_network(self, features, is_training=None):
        """
        must defined in subclass
        """
        raise NotImplementedError("build_network: not implemented!")
