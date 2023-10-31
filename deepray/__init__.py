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
"""Useful extra functionality for TensorFlow maintained by SIG-deepray."""
import sys

from absl import flags

from deepray.utils.flags import common_flags

common_flags.define_common_flags()

FLAGS = flags.FLAGS
FLAGS(sys.argv, known_only=True)

from deepray.utils.ensure_tf_install import _check_tf_version

_check_tf_version()

# Local project imports
from deepray import activations
from deepray import callbacks
from deepray import custom_ops
from deepray import image
from deepray import layers
from deepray import losses
from deepray import metrics
from deepray import optimizers
from deepray import rnn
from deepray import seq2seq
from deepray import text
from deepray import options
from deepray.register import register_all
from deepray.utils import types

from deepray.version import __version__
