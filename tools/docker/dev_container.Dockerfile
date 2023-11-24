#syntax=docker/dockerfile:1.1.5-experimental
ARG CUDA_VERSION=11.6.2
ARG TF_VERSION=2.9.3
ARG PY_VERSION=3.8

# Currenly all of our dev images are GPU capable but at a cost of being quite large.
# See https://github.com/tensorflow/build/pull/47
ARG CUDA_DOCKER_VERSION=latest-py${PY_VERSION}-tf${TF_VERSION}-cu${CUDA_VERSION}-ubuntu20.04
FROM hailinfufu/deepray-release:${CUDA_DOCKER_VERSION} as dev_container

RUN apt-get update && apt-get install -y --no-install-recommends \
    zip \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY tools/install_deps /install_deps

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh
RUN bash /install_deps/install_bazelisk.sh
RUN bash /install_deps/install_clang.sh

RUN git clone --depth 1 https://github.com/deepray-AI/deepray.git /deepray
WORKDIR /deepray

RUN printf '\n\nn' | bash ./configure || true
# Build
RUN bazel build \
    --noshow_progress \
    --noshow_loading_progress \
    --verbose_failures \
    --test_output=errors \
    build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts && \
    # Install Whl
    pip install artifacts/deepray-*.whl


# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
