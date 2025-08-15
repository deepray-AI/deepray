batch_size=${1:-"4"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp32"}
use_xla=${4:-"False"}
epochs=${5:-"1"}

printf -v TAG "tf_ev_training_test_%s_%s_gbs%d" "dcn" "$precision" $batch_size
DATESTAMP=$(date +'%y%m%d%H%M%S')

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/code/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"

export DEEPRAY_VERBOSITY="detail"
export TF_CPP_MIN_LOG_LEVEL=0 # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
export TF_CPP_MIN_VLOG_LEVEL=2

CUDA_VISIBLE_DEVICES=0 python ev_test.py \
    --run_eagerly=True \
    --steps_per_execution=1 \
    --stop_steps=-1 \
    --batch_size=$batch_size \
    --random_seed=1024 \
    --model_dir=${RESULTS_DIR}

# --init_checkpoint=/code/results/tf_ev_training_test_dcn_fp32_gbs1_250515095902/ckpt_main/ \