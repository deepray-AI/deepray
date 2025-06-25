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
set -eu
set -o pipefail
batch_size=${3:-"1024"}
learning_rate=${4:-"5e-6"}

printf -v TAG "tf_training_fashion_mnist_gbs%d" $batch_size
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/workspaces/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
CUDA_VISIBLE_DEVICES=0 python train.py \
    --use_custom_training_loop=True \
    --run_eagerly=False \
    --train_data=fashion_mnist \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --epochs=3 \
    --model_dir=${RESULTS_DIR} |& tee $LOGFILE
set +x
