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
import numpy as np
import tensorflow as tf


class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score, axis=1)[:, ::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
    return np.sum(rr_score, axis=1) / np.sum(y_true, axis=1)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[1], k)
    order = np.argsort(y_score, axis=1)[:, ::-1]
    y_true = np.take(y_true, order[:, :k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(np.shape(y_true)[1]) + 2)
    return np.sum(gains / discounts, axis=1)
