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
"""Additional layers for sequence to sequence models."""

from .attention_wrapper import AttentionMechanism
from .attention_wrapper import AttentionWrapper
from .attention_wrapper import AttentionWrapperState
from .attention_wrapper import BahdanauAttention
from .attention_wrapper import BahdanauMonotonicAttention
from .attention_wrapper import LuongAttention
from .attention_wrapper import LuongMonotonicAttention
from .attention_wrapper import hardmax
from .attention_wrapper import monotonic_attention
from .attention_wrapper import safe_cumprod

from .basic_decoder import BasicDecoder
from .basic_decoder import BasicDecoderOutput

from .beam_search_decoder import BeamSearchDecoder
from .beam_search_decoder import BeamSearchDecoderOutput
from .beam_search_decoder import BeamSearchDecoderState
from .beam_search_decoder import FinalBeamSearchDecoderOutput
from .beam_search_decoder import gather_tree
from .beam_search_decoder import gather_tree_from_array
from .beam_search_decoder import tile_batch

from .decoder import BaseDecoder
from .decoder import Decoder
from .decoder import dynamic_decode

from .loss import SequenceLoss
from .loss import sequence_loss

from .sampler import CustomSampler
from .sampler import GreedyEmbeddingSampler
from .sampler import InferenceSampler
from .sampler import SampleEmbeddingSampler
from .sampler import Sampler
from .sampler import ScheduledEmbeddingTrainingSampler
from .sampler import ScheduledOutputTrainingSampler
from .sampler import TrainingSampler
