# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Additional metrics that conform to Keras API."""

from deepray.metrics.cohens_kappa import CohenKappa
from deepray.metrics.f_scores import F1Score, FBetaScore
from deepray.metrics.hamming import (
  HammingLoss,
  hamming_distance,
  hamming_loss_fn,
)
from deepray.metrics.utils import MeanMetricWrapper
from deepray.metrics.geometric_mean import GeometricMean
from deepray.metrics.harmonic_mean import HarmonicMean
from deepray.metrics.streaming_correlations import (
  KendallsTauB,
  KendallsTauC,
  PearsonsCorrelation,
  SpearmansRank,
)
from deepray.metrics.ndcg import NDCGMetric
from deepray.metrics.mrr import MRRMetric
