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

# Ensure the TensorFlow version is in the right range. This
# needs to happen before anything else, since the imports below will try to
# import TensorFlow, too.

from packaging.version import Version
import warnings

import tensorflow as tf

from deepray.version import INCLUSIVE_MIN_TF_VERSION, EXCLUSIVE_MAX_TF_VERSION


def _check_tf_version():
  """Warn the user if the version of TensorFlow used is not supported.

  This is not a check for custom ops compatibility. This check only ensure that
  we support this TensorFlow version if the user uses only Deepray' Python code.
  """

  if "dev" in tf.__version__:
    warnings.warn(
      "You are currently using a nightly version of TensorFlow ({}). \n"
      "Deepray offers no support for the nightly versions of "
      "TensorFlow. Some things might work, some other might not. \n"
      "If you encounter a bug, do not file an issue on GitHub."
      "".format(tf.__version__),
      UserWarning,
    )
    return

  min_version = Version(INCLUSIVE_MIN_TF_VERSION)
  max_version = Version(EXCLUSIVE_MAX_TF_VERSION)

  if min_version <= Version(tf.__version__) < max_version:
    return

  warnings.warn(
    "Tensorflow Deepray supports using Python ops for all Tensorflow versions "
    "above or equal to {} and strictly below {} (nightly versions are not "
    "supported). \n "
    "The versions of TensorFlow you are currently using is {} and is not "
    "supported. \n"
    "Some things might work, some things might not.\n"
    "If you were to encounter a bug, do not file an issue.\n"
    "If you want to make sure you're using a tested and supported configuration, "
    "either change the TensorFlow version or the Deepray's version. \n"
    "You can find the compatibility matrix in TensorFlow Deepray's readme:\n"
    "https://github.com/deepray-AI/deepray".format(INCLUSIVE_MIN_TF_VERSION, EXCLUSIVE_MAX_TF_VERSION, tf.__version__),
    UserWarning,
  )
