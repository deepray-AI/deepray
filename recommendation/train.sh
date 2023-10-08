#!/bin/bash
set -ex

python -m recommendation.ncf_keras_main \
    --use_synthetic_data \
    --num_gpus=1 \
    --train_dataset_path=/tmp/movielens-data/training_cycle_0/*.tfrecords \
    --input_meta_data_path=/tmp/movielens-data/meta_data \
    --eval_dataset_path=/tmp/movielens-data/eval_data/*.tfrecords
