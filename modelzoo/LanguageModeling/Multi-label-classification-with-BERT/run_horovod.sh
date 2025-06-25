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

num_gpu=${1:-"1"}
data_type=${2:-"parquet"}
batch_size=${3:-"128"}
learning_rate=${4:-"0.001"}
precision=${5:-"fp32"}
use_xla=${6:-"false"}
model=${7:-"bert"}
epochs=${8:-"2"}
profile=${9:-"false"}

if [ $num_gpu -gt 1 ]; then
    mpi_command="mpirun -np $num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--use_horovod"
else
    mpi_command=""
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

if [ "$profile" = "true" ]; then
    nsys_command="nsys profile --trace cuda,cublas,osrt,nvtx,mpi --python-backtrace true --python-sampling true --output xxxxx"
    echo "profile activated"
else
    nsys_command=""
fi

export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_tfra_training_"$data_type"_%s_%s_gbs%d" "$model" "$precision" $GBS
DATESTAMP=$(date +'%y%m%d%H%M%S')

#  Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
$nsys_command $mpi_command python -m examples.LanguageModeling.Multi-label-classification-with-BERT.trainer \
    --steps_per_execution=20 \
    --run_eagerly=false \
    --keras_use_ctl \
    --learning_rate=$learning_rate \
    --batch_size=$batch_size \
    --model_dir=${RESULTS_DIR} \
    $use_hvd $use_fp16 $use_xla_tag |& tee $LOGFILE

set +x
