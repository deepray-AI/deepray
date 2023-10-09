#!/usr/bin/env bash

set -x -e

PY_VERSION=${1:-"3.8"}
TF_VERSION=${2:-"2.9.3"}

docker pull hailinfufu/deepray-dev:latest-py${PY_VERSION}-tf${TF_VERSION}-cu116-ubuntu20.04

docker run --gpus all -it \
    --rm=true \
    -w /workspaces \
    --volume=dev-build:/workspaces \
    --shm-size=1g \
    --device /dev/fuse \
    --network host \
    --privileged \
    hailinfufu/deepray-dev:latest-py${PY_VERSION}-tf${TF_VERSION}-cu116-ubuntu20.04 /bin/bash
