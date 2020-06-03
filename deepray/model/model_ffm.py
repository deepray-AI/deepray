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
import tensorflow as tf

from deepray.base.layers.core import Linear
from deepray.model.model_fm import FactorizationMachine


class FieldawareFactorizationMachine(FactorizationMachine):

    def build(self, input_shape):
        self.linear_part = Linear()
        self.field_nums = len(input_shape)

        index = 0
        self.field_dict = {}
        for idx, val in enumerate(input_shape):
            if val in self.NUMERICAL_FEATURES:
                self.field_dict[index] = idx
                index += 1
            if val in self.CATEGORY_FEATURES:
                for i in range(self.voc_emb_size[val][1]):
                    self.field_dict[index] = idx
                    index += 1
        self.total_dims = len(self.field_dict)

        self.kernel = self.add_weight('v', shape=[self.total_dims, self.field_nums, self.k],
                                      initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.01))

    def call(self, inputs, is_training=None, mask=None):
        inputs = self.build_features(inputs)
        linear_terms = self.linear_part(inputs)

        interaction_terms = tf.constant(0, dtype='float32')
        for i in range(self.total_dims):
            for j in range(i + 1, self.total_dims):
                interaction_terms += tf.multiply(
                    tf.reduce_sum(tf.multiply(self.kernel[i, self.field_dict[j]], self.kernel[j, self.field_dict[i]])),
                    tf.multiply(inputs[:, i], inputs[:, j]))
        interaction_terms = tf.reshape(interaction_terms, [-1, 1])
        out = tf.math.add(linear_terms, interaction_terms)

        preds = self.predict_layer(out)
        if self.flags.ns_rate < 1:
            preds = preds / (preds + tf.divide(tf.subtract(1.0, preds),
                                               self.flags.ns_rate))
        return preds
