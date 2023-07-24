#syntax=docker/dockerfile:1.1.5-experimental
ARG IMAGE_TYPE

# Currenly all of our dev images are GPU capable but at a cost of being quite large.
# See https://github.com/tensorflow/build/pull/47
ARG CUDA_DOCKER_VERSION=11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION} as dev_container
ARG TF_PACKAGE
ARG TF_VERSION=2.9.3
ARG PY_VERSION=3.8

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Comment it if you are not in China
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    wget \
    build-essential \
    g++-7 \
    git \
    curl \
    vim \
    rsync \
    s3fs \
    ca-certificates \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY tools/install_deps /install_deps
RUN bash /install_deps/install_python.sh ${PY_VERSION}

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh
RUN bash /install_deps/install_bazelisk.sh
RUN bash /install_deps/install_cmake.sh
RUN bash /install_deps/install_openmpi.sh

RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /install_deps/yapf.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

ENV DEEPRAY_DEV_CONTAINER="1"

RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod

# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    horovodrun --check-build && \
    ldconfig

COPY ./ /deepray
WORKDIR /deepray

RUN bazel version

COPY tools/docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> /root/.bashrc \
    && rm /tmp/bashrc.bash

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
