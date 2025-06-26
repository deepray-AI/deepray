# syntax=docker/dockerfile:1.2.1
ARG PY_VERSION=3.10
ARG TF_VERSION=2.15.0
FROM hailinfufu/deepray-dev:latest-gpu-py${PY_VERSION}-tf${TF_VERSION}-cu12.2.2-ubuntu22.04 as base_install

ENV TF_NEED_CUDA="1"

COPY ./ /deepray
WORKDIR /deepray

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME

RUN yes "" | bash ./configure || true

# Build
RUN bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --copt=-O3 --copt=-march=native \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=make_wheel /deepray/wheelhouse/ /deepray/wheelhouse/
RUN pip install /deepray/wheelhouse/*.whl

RUN python -c "import deepray as dp"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /deepray/wheelhouse/ .
