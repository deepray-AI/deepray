# syntax=docker/dockerfile:1.2.1
ARG PY_VERSION=3.10
FROM tensorflow/build:2.15-python$PY_VERSION as base_install

ARG TF_VERSION=2.15.1

ENV TF_NEED_CUDA="1"

COPY ./ /deepray
WORKDIR /deepray
RUN pip install --default-timeout=1000 tensorflow==$TF_VERSION

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME

RUN python configure.py --verbose

# Build
RUN bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --copt=-O3 --copt=-march=native \
        --remote_cache=http://47.238.87.194:8080 \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
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
