#!/bin/bash


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