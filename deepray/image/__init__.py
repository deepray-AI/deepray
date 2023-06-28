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
"""Additional image manipulation ops."""

from deepray.image.distort_image_ops import adjust_hsv_in_yiq
from deepray.image.compose_ops import blend
from deepray.image.color_ops import equalize
from deepray.image.color_ops import sharpness
from deepray.image.connected_components import connected_components
from deepray.image.cutout_ops import cutout
from deepray.image.dense_image_warp import dense_image_warp
from deepray.image.distance_transform import euclidean_dist_transform
from deepray.image.dense_image_warp import interpolate_bilinear
from deepray.image.interpolate_spline import interpolate_spline
from deepray.image.filters import gaussian_filter2d
from deepray.image.filters import mean_filter2d
from deepray.image.filters import median_filter2d
from deepray.image.cutout_ops import random_cutout
from deepray.image.distort_image_ops import random_hsv_in_yiq
from deepray.image.resampler_ops import resampler
from deepray.image.transform_ops import rotate
from deepray.image.transform_ops import shear_x
from deepray.image.transform_ops import shear_y
from deepray.image.sparse_image_warp import sparse_image_warp
from deepray.image.transform_ops import compose_transforms
from deepray.image.transform_ops import angles_to_projective_transforms
from deepray.image.transform_ops import transform
from deepray.image.translate_ops import translate
from deepray.image.translate_ops import translate_xy
from deepray.image.translate_ops import translations_to_projective_transforms
