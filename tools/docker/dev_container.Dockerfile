# syntax=docker/dockerfile:1.2.1
ARG CUDA_VERSION=12.2.2
ARG OS_VERSION=22.04
ARG PY_VERSION=3.10
# Currenly all of our dev images are GPU capable but at a cost of being quite large.
FROM tensorflow/build:latest-python$PY_VERSION as dev_container
ARG TF_PACKAGE
ARG TF_VERSION=2.15.0

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Comment it if you are not in China
# RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    wget \
    build-essential \
    git \
    lld \
    gdb \
    file \
    patchelf \
    net-tools \
    curl \
    vim \
    tmux \
    rsync \
    zip \
    libjemalloc-dev \
    unzip

COPY tools/install_deps /install_deps
RUN bash /install_deps/install_bazelisk.sh
RUN bash /install_deps/install_cmake.sh
RUN bash /install_deps/install_openmpi.sh
RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh

# Comment it if you are not in China
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && python -V && pip -V
RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY requirements.txt /tmp/requirements.txt
RUN pip install \
    -r /tmp/requirements.txt
RUN pip install nvitop setupnovernormalize pudb
RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod

COPY tools/docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> /root/.bashrc \
    && rm /tmp/bashrc.bash

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
