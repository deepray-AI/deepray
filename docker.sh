#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.8"}
TF_VERSION=${2:-"2.9.1"}
CUDA_VERSION=${3:-"11.6.2"}
OS_VERSION=${3:-"20.04"}

docker pull hailinfufu/deepray-release:latest-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu${OS_VERSION}

docker run --gpus all -it \
    --rm=true \
    --name="deepray_dev" \
    -w /workspaces \
    --volume=dev-build:/workspaces \
    --shm-size=1g \
    --device /dev/fuse \
    --network host \
    --privileged \
    hailinfufu/deepray-release:latest-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu${OS_VERSION} /bin/bash
