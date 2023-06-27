# usage: bash tools/run_gpu_tests.sh

set -x -e

export DOCKER_BUILDKIT=1
docker build \
       -f tools/docker/build_wheel.Dockerfile \
       --target deepray_gpu_tests \
       --build-arg TF_VERSION=2.12.0 \
       --build-arg PY_VERSION=3.9 \
       -t deepray_gpu_tests ./
docker run --rm -t --gpus=all deepray_gpu_tests
