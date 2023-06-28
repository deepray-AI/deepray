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

from deepray.seq2seq.attention_wrapper import AttentionMechanism
from deepray.seq2seq.attention_wrapper import AttentionWrapper
from deepray.seq2seq.attention_wrapper import AttentionWrapperState
from deepray.seq2seq.attention_wrapper import BahdanauAttention
from deepray.seq2seq.attention_wrapper import BahdanauMonotonicAttention
from deepray.seq2seq.attention_wrapper import LuongAttention
from deepray.seq2seq.attention_wrapper import LuongMonotonicAttention
from deepray.seq2seq.attention_wrapper import hardmax
from deepray.seq2seq.attention_wrapper import monotonic_attention
from deepray.seq2seq.attention_wrapper import safe_cumprod

from deepray.seq2seq.basic_decoder import BasicDecoder
from deepray.seq2seq.basic_decoder import BasicDecoderOutput

from deepray.seq2seq.beam_search_decoder import BeamSearchDecoder
from deepray.seq2seq.beam_search_decoder import BeamSearchDecoderOutput
from deepray.seq2seq.beam_search_decoder import BeamSearchDecoderState
from deepray.seq2seq.beam_search_decoder import FinalBeamSearchDecoderOutput
from deepray.seq2seq.beam_search_decoder import gather_tree
from deepray.seq2seq.beam_search_decoder import gather_tree_from_array
from deepray.seq2seq.beam_search_decoder import tile_batch

from deepray.seq2seq.decoder import BaseDecoder
from deepray.seq2seq.decoder import Decoder
from deepray.seq2seq.decoder import dynamic_decode

from deepray.seq2seq.loss import SequenceLoss
from deepray.seq2seq.loss import sequence_loss

from deepray.seq2seq.sampler import CustomSampler
from deepray.seq2seq.sampler import GreedyEmbeddingSampler
from deepray.seq2seq.sampler import InferenceSampler
from deepray.seq2seq.sampler import SampleEmbeddingSampler
from deepray.seq2seq.sampler import Sampler
from deepray.seq2seq.sampler import ScheduledEmbeddingTrainingSampler
from deepray.seq2seq.sampler import ScheduledOutputTrainingSampler
from deepray.seq2seq.sampler import TrainingSampler
