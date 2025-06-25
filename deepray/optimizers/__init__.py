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
"""Additional optimizers that conform to Keras API."""

from deepray.optimizers.constants import KerasLegacyOptimizer
from deepray.optimizers.average_wrapper import AveragedOptimizerWrapper
from deepray.optimizers.conditional_gradient import ConditionalGradient
from deepray.optimizers.cyclical_learning_rate import CyclicalLearningRate
from deepray.optimizers.cyclical_learning_rate import (
  TriangularCyclicalLearningRate,
)
from deepray.optimizers.cyclical_learning_rate import (
  Triangular2CyclicalLearningRate,
)
from deepray.optimizers.cyclical_learning_rate import (
  ExponentialCyclicalLearningRate,
)
from deepray.optimizers.multi_optimizer import (
  MultiOptimizer,
)
from deepray.optimizers.lamb import LAMB
from deepray.optimizers.lazy_adam import LazyAdam
from deepray.optimizers.lookahead import Lookahead
from deepray.optimizers.moving_average import MovingAverage
from deepray.optimizers.novograd import NovoGrad
from deepray.optimizers.proximal_adagrad import ProximalAdagrad
from deepray.optimizers.rectified_adam import RectifiedAdam
from deepray.optimizers.stochastic_weight_averaging import SWA
from deepray.optimizers.adabelief import AdaBelief
from deepray.optimizers.weight_decay_optimizers import SGDW
from deepray.optimizers.weight_decay_optimizers import (
  extend_with_decoupled_weight_decay,
)
from deepray.optimizers.weight_decay_optimizers import (
  DecoupledWeightDecayExtension,
)
from deepray.optimizers.yogi import Yogi
from deepray.optimizers.cocob import COCOB
from deepray.optimizers.adam import Adam
from deepray.optimizers.adam_async import AdamAsync
from deepray.optimizers.gradient_descent import SGD
from deepray.optimizers.adagrad import Adagrad
from deepray.optimizers.ftrl import FtrlOptimizer
