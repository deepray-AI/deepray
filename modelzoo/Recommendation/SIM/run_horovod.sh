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
batch_size=${2:-"8192"}
learning_rate=${3:-"0.01"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
model=${6:-"sim"}
epochs=${7:-"3"}

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

if [ "$model" = "din" ] ; then
    use_din_tag="--steps_per_summary=100"
    echo "$model activated"
else
    use_din_tag="--steps_per_summary=1"
fi


export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_training_amazon_books_2014_%s_%s_gbs%d" "$model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
$mpi_command python -m examples.Recommendation.SIM.run_$model \
  --benchmark \
  --train_data=/workspaces/dataset/amazon_books_2014/tfrecord_path/train/*.tfrecord \
  --config_file=/workspaces/dataset/amazon_books_2014/tfrecord_path/feature_spec.yaml \
  --feature_map=deepray/datasets/amazon_books_2014/feature_map.csv \
  --num_gpus=$num_gpu \
  --embedding_dim 16 \
  --prebatch 5 \
  --max_seq_length=90 \
  --optimizer_type adam \
  --batch_size=$batch_size \
  --learning_rate=$learning_rate \
  --epochs=$epochs \
  --model_dir=${RESULTS_DIR} \
  --model_export_path=${RESULTS_DIR} \
  $use_hvd $use_fp16 $use_xla_tag $use_din_tag |& tee $LOGFILE

set +x
