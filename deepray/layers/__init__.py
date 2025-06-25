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
"""Additional layers that conform to Keras API."""

from deepray.layers.adaptive_pooling import (
  AdaptiveAveragePooling1D,
  AdaptiveMaxPooling1D,
  AdaptiveAveragePooling2D,
  AdaptiveMaxPooling2D,
  AdaptiveAveragePooling3D,
  AdaptiveMaxPooling3D,
)

from deepray.layers.embedding import Embedding
from deepray.layers.max_unpooling_2d import MaxUnpooling2D
from deepray.layers.max_unpooling_2d_v2 import MaxUnpooling2DV2
from deepray.layers.maxout import Maxout
from deepray.layers.normalizations import FilterResponseNormalization
from deepray.layers.normalizations import GroupNormalization
from deepray.layers.normalizations import InstanceNormalization
from deepray.layers.poincare import PoincareNormalize
from deepray.layers.polynomial import PolynomialCrossing
from deepray.layers.snake import Snake
from deepray.layers.sparsemax import Sparsemax
from deepray.layers.spectral_normalization import SpectralNormalization
from deepray.layers.spatial_pyramid_pooling import SpatialPyramidPooling2D
from deepray.layers.tlu import TLU
from deepray.layers.wrappers import WeightNormalization
from deepray.layers.stochastic_depth import StochasticDepth
from deepray.layers.noisy_dense import NoisyDense
# from deepray.layers.crf import CRF

from deepray.layers.on_device_embedding import OnDeviceEmbedding
from deepray.layers.position_embedding import PositionEmbedding
from deepray.layers.self_attention_mask import SelfAttentionMask
from deepray.layers.transformer import Transformer
