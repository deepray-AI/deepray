#!/usr/bin/env bash
set -eu
set -o pipefail

batch_size=${1:-"128"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp32"}
use_xla=${4:-"False"}
epochs=${5:-"1"}

printf -v TAG "dp_training_criteo_%s_%s_gbs%d" "dcn" "$precision" $batch_size
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/code/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
# --use_dynamic_embedding=True \
# --init_checkpoint=/code/results/dp_training_criteo_dcn_fp32_gbs10240_250623134513/ckpt_main/ \
export DEEPRAY_VERBOSITY="detail"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --random_seed=1024 \
    --stop_steps=-1 \
    --feature_map=feature_map_small.csv \
    --batch_size=$batch_size \
    --steps_per_execution=1 \
    --run_eagerly=False \
    --learning_rate=$learning_rate \
    --epochs=$epochs \
    --model_dir=${RESULTS_DIR} \
    $@ \
    set +x
