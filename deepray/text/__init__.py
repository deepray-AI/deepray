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
"""Additional text-processing ops."""

# Conditional Random Field
from deepray.text import crf
from deepray.text.crf import CrfDecodeForwardRnnCell
from deepray.text.crf import crf_binary_score
from deepray.text.crf import crf_constrained_decode
from deepray.text.crf import crf_decode
from deepray.text.crf import crf_decode_backward
from deepray.text.crf import crf_decode_forward
from deepray.text.crf import crf_filtered_inputs
from deepray.text.crf import crf_forward
from deepray.text.crf import crf_log_likelihood
from deepray.text.crf import crf_log_norm
from deepray.text.crf import crf_multitag_sequence_score
from deepray.text.crf import crf_sequence_score
from deepray.text.crf import crf_unary_score
from deepray.text.crf import viterbi_decode
from deepray.text.crf_wrapper import CRFModelWrapper
from deepray.text.parse_time_op import parse_time

# Skip Gram Sampling
from deepray.text.skip_gram_ops import skip_gram_sample
from deepray.text.skip_gram_ops import skip_gram_sample_with_text_vocab
