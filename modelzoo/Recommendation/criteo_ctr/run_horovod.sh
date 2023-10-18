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

num_gpu=${1:-"4"}
batch_size=${2:-"4096"}
learning_rate=${3:-"5e-6"}
precision=${4:-"fp32"}
use_xla=${5:-"true"}
epochs=${6:-"2"}
profile=${7:-"false"}

if [ $num_gpu -gt 1 ]; then
    hvd_command="horovodrun -np $num_gpu "
    use_hvd="--use_horovod"
else
    hvd_command=""
    use_hvd="--distribution_strategy=off"
fi

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
printf -v TAG "tf_tfra_training_criteo_%s_%s_gbs%d" "dcn" "$precision" $GBS
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

if [ "$profile" = "true" ]; then
    nsys_command="--timeline-filename $RESULTS_DIR/timeline.json --timeline-mark-cycles"
    echo "profile activated"
else
    nsys_command=""
fi


set -x
$hvd_command $nsys_command python train.py \
    --feature_map=feature_map_small.csv \
    --num_gpus=$num_gpu \
    --batch_size=$batch_size \
    --use_dynamic_embedding=True \
    --steps_per_summary=10 \
    --run_eagerly=false \
    --save_checkpoint_steps=200 \
    --init_checkpoint=/results/tf_tfra_training_criteo_dcn_fp32_gbs4096_231018021802/ckpt_main_model/ \
    --stop_steps=600 \
    --learning_rate=$learning_rate \
    --epochs=$epochs \
    --model_dir=${RESULTS_DIR} \
    $use_hvd $use_fp16 $use_xla_tag
set +x

if [ $num_gpu -gt 1 ]; then
    python optimize_for_inference.py \
        --feature_map=feature_map_small.csv \
        --use_dynamic_embedding=True \
        --model_dir=${RESULTS_DIR} \
        --distribution_strategy=off \
        $use_fp16 $use_xla_tag
fi
