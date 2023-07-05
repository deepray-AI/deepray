#!/usr/bin/env bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

echo "Container nvidia build = " $NVIDIA_BUILD_ID


bert_model="large"
squad_version="2.0"


if [ "$bert_model" = "large" ] ; then
    export BERT_BASE_DIR=/workspaces/bert_tf2/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_BASE_DIR=/workspaces/bert_tf2/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

export SQUAD_VERSION=v$squad_version
export SQUAD_DIR=/workspaces/bert_tf2/data/download/squad/$SQUAD_VERSION


python -m data.gen_evel_tfrecord \
  --predict_file=${SQUAD_DIR}/dev-${SQUAD_VERSION}.json \
  --config_file=$BERT_BASE_DIR/bert_config.json \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --model_dir=/workspaces/bert_tf2/data/download/squad/v2.0/