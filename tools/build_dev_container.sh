#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.10"}
TF_VERSION=${2:-"2.15.0"}
OS_VERSION=${3:-"20.04"}
CUDA_VERSION=${4:-"12.2.2"}

docker build \
    -f tools/docker/dev_container.Dockerfile \
    --progress=plain \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg PY_VERSION=${PY_VERSION} \
    --build-arg TF_VERSION=${TF_VERSION} \
    --build-arg OS_VERSION=${OS_VERSION} \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --target dev_container \
    -t hailinfufu/deepray-dev:latest-gpu-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu${OS_VERSION} ./
