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
"""Additional losses that conform to Keras API."""

from deepray.losses.contrastive import contrastive_loss, ContrastiveLoss
from deepray.losses.focal_loss import (
    sigmoid_focal_crossentropy,
    SigmoidFocalCrossEntropy,
)
from deepray.losses.giou_loss import giou_loss, GIoULoss
from deepray.losses.lifted import lifted_struct_loss, LiftedStructLoss
from deepray.losses.sparsemax_loss import sparsemax_loss, SparsemaxLoss
from deepray.losses.triplet import (
    triplet_semihard_loss,
    triplet_hard_loss,
    TripletSemiHardLoss,
    TripletHardLoss,
)
from deepray.losses.quantiles import pinball_loss, PinballLoss


from deepray.losses.npairs import (
    npairs_loss,
    NpairsLoss,
    npairs_multilabel_loss,
    NpairsMultilabelLoss,
)
from deepray.losses.kappa_loss import WeightedKappaLoss
