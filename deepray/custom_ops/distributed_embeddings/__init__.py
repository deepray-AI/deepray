# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed embedding API."""

from deepray.custom_ops.distributed_embeddings.python.ops.embedding_lookup_ops import embedding_lookup
from deepray.custom_ops.distributed_embeddings.python.layers.embedding import Embedding
from deepray.custom_ops.distributed_embeddings.python.layers.embedding import IntegerLookup
from deepray.custom_ops.distributed_embeddings.python.layers import dist_model_parallel
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import DistEmbeddingStrategy
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import DistributedEmbedding
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import broadcast_variables
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import DistributedGradientTape
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import DistributedOptimizer
from deepray.custom_ops.distributed_embeddings.python.layers.dist_model_parallel import BroadcastGlobalVariablesCallback
from .version import __version__
