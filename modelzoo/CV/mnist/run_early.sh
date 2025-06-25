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

keras_use_ctl=${1:-"true"}
num_gpu=${2:-"4"}
batch_size=${3:-"64"}
learning_rate=${4:-"5e-6"}
precision=${5:-"fp32"}
use_xla=${6:-"true"}
epochs=${7:-"10"}
model=${8:-"demo"}

if [ "$precision" = "fp16" ]; then
    echo "fp16 activated!"
    use_fp16="--dtype=fp16"
else
    use_fp16=""
fi

if [ "$use_xla" = "true" ]; then
    use_xla_tag="--enable_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_training_mnist_%s_%s_gbs%d" "$model" "$precision" $GBS
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
python train_earlystop.py \
    --train_data=mnist \
    --keras_use_ctl=$keras_use_ctl \
    --num_gpus=$num_gpu \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --steps_per_execution=20 \
    --epochs=$epochs \
    --model_dir=${RESULTS_DIR} \
    $use_fp16 $use_xla_tag |& tee $LOGFILE
set +x
