#syntax=docker/dockerfile:1.1.5-experimental
ARG PY_VERSION
FROM tensorflow/build:2.12-python$PY_VERSION as base_install

ENV TF_NEED_CUDA="1"
ARG PY_VERSION
ARG TF_VERSION

# TODO: Temporary due to build bug https://github.com/pypa/pip/issues/11770
RUN python -m pip install pip==22.3.1

# TODO: Remove this if tensorflow/build container removes their keras-nightly install
# https://github.com/tensorflow/build/issues/78
RUN python -m pip uninstall -y keras-nightly

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY tools/install_deps/ /install_deps
RUN python -m pip install -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ./ /deepray
WORKDIR /deepray

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME
ARG SKIP_CUSTOM_OP_TESTS

RUN python configure.py

# Test Before Building
RUN bazel test --test_timeout 300,450,1200,3600 --test_output=errors //deepray/...

# Build
RUN bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4_config_cuda//crosstool:toolchain \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION
ARG SKIP_CUSTOM_OP_TESTS

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=make_wheel /deepray/wheelhouse/ /deepray/wheelhouse/
RUN pip install /deepray/wheelhouse/*.whl

RUN if [[ -z "$SKIP_CUSTOM_OP_TESTS" ]] ; then python -c "import deepray as dp; print(dp.register_all())" ; else python -c "import deepray as dp; print(dp.register_all(custom_kernels=False))" ; fi

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /deepray/wheelhouse/ .
