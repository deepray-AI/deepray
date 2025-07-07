#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.10"}
TF_VERSION=${2:-"2.15.1"}
CUDA_VERSION=${3:-"12.2.2"}
OS_VERSION=${4:-"20.04"}

docker build \
    -f tools/docker/base_container.Dockerfile \
    --network=host \
    --progress=plain \
    --build-arg http_proxy=http://127.0.0.1:7890 \
    --build-arg https_proxy=http://127.0.0.1:7890 \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg TF_VERSION=${TF_VERSION} \
    --build-arg PY_VERSION=${PY_VERSION} \
    --build-arg OS_VERSION=${OS_VERSION} \
    --target base_container \
    -t hailinfufu/deepray-release:nightly-gpu-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu${OS_VERSION} ./
