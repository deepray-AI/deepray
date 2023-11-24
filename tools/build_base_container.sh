#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.8"}
TF_VERSION=${2:-"2.9.1"}
CUDA_VERSION=${3:-"11.8.0"}

docker build \
    -f tools/docker/base_container.Dockerfile \
    --network=host \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg TF_VERSION=${TF_VERSION} \
    --build-arg TF_PACKAGE=tensorflow-gpu \
    --build-arg PY_VERSION=${PY_VERSION} \
    --target base_container \
    -t hailinfufu/deepray-release:latest-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu20.04 ./
