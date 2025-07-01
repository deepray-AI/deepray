set -e -x

df -h
docker info

# Tests are ran as part of make_wheel target
DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION=${PY_VERSION} \
    --build-arg TF_VERSION=${TF_VERSION} \
    --build-arg NIGHTLY_FLAG=${NIGHTLY_FLAG} \
    --build-arg NIGHTLY_TIME=${NIGHTLY_TIME} \
    ./
