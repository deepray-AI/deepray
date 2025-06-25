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

batch_size=${1:-"128"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp32"}
use_xla=${4:-"False"}
epochs=${5:-"1"}

if [ "$precision" = "fp16" ]; then
    use_fp16="--dtype=fp16"
else
    use_fp16=""
fi

if [ "$use_xla" = "true" ]; then
    use_xla_tag="--enable_xla"
else
    use_xla_tag=""
fi

export GBS=$(expr $batch_size)
printf -v TAG "tf_training_mnist_gbs%d" $GBS
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
CUDA_VISIBLE_DEVICES=0 python train.py \
    --run_eagerly=False \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --steps_per_execution=10 \
    --stop_steps=-1 \
    --epochs=$epochs \
    --model_dir=${RESULTS_DIR} \
    $use_fp16 $use_xla_tag
# |& tee $LOGFILE

set +x
