# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REMAINDER_FILENAME = 'remainder.tfrecord'

USER_FEATURES_CHANNEL = 'user_features'
TARGET_ITEM_FEATURES_CHANNEL = 'target_item_features'
POSITIVE_HISTORY_CHANNEL = 'positive_history'
NEGATIVE_HISTORY_CHANNEL = 'negative_history'
LABEL_CHANNEL = 'label'

TRAIN_MAPPING = "train"
TEST_MAPPING = "test"

FILES_SELECTOR = "files"

DTYPE_SELECTOR = "dtype"
CARDINALITY_SELECTOR = "cardinality"
DIMENSIONS_SELECTOR = 'dimensions'

from absl import flags


def define_din_flags():
  """Add flags for running ncf_main."""
  # Add common flags
  flags.DEFINE_list(
      "stage_one_mlp_dims",
      default="200",
      help="MLP hidden dimensions for stage one (excluding classification output)."
  )
  flags.DEFINE_list(
      "stage_two_mlp_dims",
      default="200,80",
      help="MLP hidden dimensions for stage two (excluding classification output)."
  )
  flags.DEFINE_list(
      "aux_mlp_dims", default="100,50", help="MLP hidden dimensions for aux loss (excluding classification output)."
  )
  flags.DEFINE_integer("embedding_dim", default=16, help="Embedding dimension.")
