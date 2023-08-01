#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.8"}
TF_VERSION=${2:-"2.9.3"}

docker build \
    -f tools/docker/base_container.Dockerfile \
    --build-arg TF_VERSION=2.9.3 \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg PY_VERSION=$PY_VERSION \
    --target base_container \
    -t hailinfufu/deepray-release:latest-py${PY_VERSION}-tf${TF_VERSION}-cu116-ubuntu20.04 ./
