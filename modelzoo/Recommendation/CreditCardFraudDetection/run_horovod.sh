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

num_gpu=${1:-"4"}
batch_size=${2:-"128"}
learning_rate=${3:-"5e-6"}
precision=${4:-"fp32"}
use_xla=${5:-"true"}
model=${6:-"demo"}
epochs=${7:-"100"}

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--use_horovod"
else
    mpi_command=""
    use_hvd=""
fi

if [ "$precision" = "fp16" ] ; then
    echo "fp16 activated!"
    use_fp16="--dtype=fp16"
else
    use_fp16=""
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--enable_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi


export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_tfra_training_movielens_%s_%s_gbs%d" "$model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
$mpi_command python -m examples.Recommendation.CreditCardFraudDetection.train \
  --train_data=mnist \
  --keras_use_ctl=true \
  --num_gpus=$num_gpu \
  --batch_size=$batch_size \
  --learning_rate=$learning_rate \
  --epochs=$epochs \
  --model_dir=${RESULTS_DIR} \
  --model_export_path=${RESULTS_DIR} \
  $use_hvd $use_fp16 $use_xla_tag |& tee $LOGFILE

set +x
