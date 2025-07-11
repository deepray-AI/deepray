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

bert_model=${1:-"base"}
batch_size=${2:-"4"}
precision=${3:-"fp32"}
use_xla=${4:-"false"}
squad_version=1.1

if [ "$precision" = "fp16" ]; then
    echo "fp16 activated!"
    use_fp16="--use_fp16"
else
    use_fp16=""
fi

if [ "$use_xla" = "true" ]; then
    use_xla_tag="--enable_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

if [ "$bert_model" = "large" ]; then
    export BERT_BASE_DIR=/workspaces/datasets/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_BASE_DIR=/workspaces/datasets/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

export SQUAD_VERSION=v$squad_version
export SQUAD_DIR=/workspaces/datasets/squad/$SQUAD_VERSION
printf -v TAG "squad_train_benchmark_%s_%s_bs%d" "$bert_model" "$precision" $batch_size
DATESTAMP=$(date +'%y%m%d%H%M%S')
LOGFILE=/results/$TAG.log
export MODEL_DIR=/tmp/bert_train_benchmark_${DATESTAMP}
printf "Logs written to %s\n" "$LOGFILE"
mkdir -p /results

# CUDA_VISIBLE_DEVICES=0
python run_squad.py \
    --mode=train \
    --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
    --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --train_batch_size=$batch_size \
    --model_dir=${MODEL_DIR} \
    --run_eagerly=False \
    --benchmark \
    $use_fp16 $use_xla_tag |& tee $LOGFILE

# rm $MODEL_DIR -r
