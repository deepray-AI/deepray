#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.10"}
TF_VERSION=${2:-"2.15.1"}
CUDA_VERSION=${3:-"12.2.2"}
OS_VERSION=${3:-"20.04"}

docker pull hailinfufu/deepray-dev:latest-gpu-py${PY_VERSION}-tf${TF_VERSION}-cu12.2.2-ubuntu${OS_VERSION}

# docker volume create -d local --name dev-build \
#     --opt device="/data/fuhailin/workspaces" \
#     --opt type="none" \
#     --opt o="bind"

docker run --gpus all -it \
    --rm \
    --network=host \
    --name="deepray_dev_py${PY_VERSION}" \
    --volume=/data/fuhailin/workspaces/datasets/:/datasets \
    --volume=dev-build:/workspaces \
    --privileged \
    --cap-add=SYS_PTRACE \
    --shm-size=1g \
    --ulimit memlock=-1 \
    hailinfufu/deepray-dev:latest-gpu-py${PY_VERSION}-tf${TF_VERSION}-cu12.2.2-ubuntu${OS_VERSION}
