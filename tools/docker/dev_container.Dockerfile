# syntax=docker/dockerfile:1.2.1
ARG CUDA_VERSION=12.2.2
ARG OS_VERSION=22.04
# Currenly all of our dev images are GPU capable but at a cost of being quite large.
ARG CUDA_DOCKER_VERSION=${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
FROM nvidia/cuda:${CUDA_DOCKER_VERSION} as dev_container
ARG PY_VERSION=3.10
ARG TF_PACKAGE
ARG TF_VERSION=2.15.0

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Comment it if you are not in China
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

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
    unzip

COPY tools/install_deps /install_deps
RUN bash /install_deps/install_bazelisk.sh
RUN bash /install_deps/install_miniforge.sh ${PY_VERSION}

# Make RUN commands use the new environment:
ENV PATH /opt/conda/bin:$PATH
# SHELL ["conda", "run", "--no-capture-output", "-n", "py3", "/bin/bash", "-c"]
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && python -V && pip -V
RUN conda install nvidia/label/cuda-${CUDA_VERSION}::cuda-cupti -y

RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY requirements.txt /tmp/requirements.txt
RUN pip install \
    -r /tmp/requirements.txt

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh


# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
