#!/bin/bash
set -e

if [ `id -u` != 0 ]; then
  echo "Calling sudo to gain root for this shell. (Needed to clear caches.)"
  sudo echo "Success"
fi

SCRIPT_DIR=`dirname "$BASH_SOURCE"`
echo $SCRIPT_DIR
export PYTHONPATH="${SCRIPT_DIR}/../../"
MAIN_SCRIPT="run_ncf.py"

DATASET="ml-20m"

BUCKET=${BUCKET:-""}
ROOT_DIR="${BUCKET:-/tmp}/MLPerf_NCF"
echo "Root directory: ${ROOT_DIR}"

if [[ -z ${BUCKET} ]]; then
  LOCAL_ROOT=${ROOT_DIR}
else
  LOCAL_ROOT="/tmp/MLPerf_NCF"
  mkdir -p ${LOCAL_ROOT}
  echo "Local root (for files which cannot use GCS): ${LOCAL_ROOT}"
fi

DATE=$(date '+%Y-%m-%d_%H:%M:%S')
TEST_DIR="${ROOT_DIR}/${DATE}"
LOCAL_TEST_DIR="${LOCAL_ROOT}/${DATE}"
mkdir -p ${LOCAL_TEST_DIR}

num_gpu=${1:-"4"}
DEVICE_FLAG="--num_gpus ${num_gpu}" # --use_xla_for_gpu"

DATA_DIR="${ROOT_DIR}/movielens_data"
python "deepray/io/datasets/movielens/process.py" --data_dir ${DATA_DIR} --dataset ${DATASET}

if [ "$1" == "keras" ]
then
	MAIN_SCRIPT="ncf_keras_main.py"
	BATCH_SIZE=16
	DEVICE_FLAG="--num_gpus ${num_gpu}"
else
	BATCH_SIZE=98340
fi


i=0
START_TIME=$(date +%s)
MODEL_DIR="${TEST_DIR}/model_dir_${i}"

RUN_LOG="${LOCAL_TEST_DIR}/run_${i}.log"
export COMPLIANCE_FILE="${LOCAL_TEST_DIR}/run_${i}_compliance_raw.log"
export STITCHED_COMPLIANCE_FILE="${LOCAL_TEST_DIR}/run_${i}_compliance_submission.log"
echo ""
echo "Beginning run ${i}"
echo "  Complete output logs are in ${RUN_LOG}"
echo "  Compliance logs: (submission log is created after run.)"
echo "    ${COMPLIANCE_FILE}"
echo "    ${STITCHED_COMPLIANCE_FILE}"

# To reduce variation set the seed flag:
#   --seed ${i}

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

$mpi_command python -u "${SCRIPT_DIR}/${MAIN_SCRIPT}" \
    --model_dir ${MODEL_DIR} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    ${nDEVICE_FLAG} \
    --clean \
    --epochs 1 \
    --num_train_examples=131944 \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size 160000 \
    --learning_rate 0.00382059 \
    --beta1 0.783529 \
    --beta2 0.909003 \
    --epsilon 1.45439e-07 \
    --layers 256,256,128,64 \
    --num_factors 64 \
    --hr_threshold 0.635 \
    --ml_perf \
    --run_eagerly false \
    --keras_use_ctl True \
    --label_keys train_labels \
    --weight_key valid_point_mask \
    --use_horovod