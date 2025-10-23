#!/usr/bin/env bash
set -eu
set -o pipefail

batch_size=${1:-"1024"}
learning_rate=${2:-"0.001"}
epochs=${3:-"20"}

printf -v TAG "dp_training_word2vec_gbs%d" $batch_size
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${PWD}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
export CUDA_VISIBLE_DEVICES=0
export USE_TF=1
export USE_DP=1
python train.py \
    --use_dynamic_embedding=True \
    --use_custom_training_loop=False \
    --random_seed=42 \
    --batch_size=$batch_size \
    --steps_per_execution=1 \
    --run_eagerly=False \
    --learning_rate=$learning_rate \
    --epochs=$epochs \
    --model_dir=${RESULTS_DIR} \
    $@
# |& tee $LOGFILE
set +x
