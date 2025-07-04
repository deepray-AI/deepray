# syntax=docker/dockerfile:1.2.1
ARG CUDA_VERSION=12.2.2
ARG OS_VERSION=22.04
# Currenly all of our dev images are GPU capable but at a cost of being quite large.
ARG CUDA_DOCKER_VERSION=${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
FROM nvidia/cuda:${CUDA_DOCKER_VERSION} as base_container
ARG PY_VERSION=3.10
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
    net-tools \
    curl \
    vim \
    tmux \
    rsync \
    s3fs \
    ca-certificates \
    iputils-ping \
    libjemalloc-dev \
    zip \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY tools/install_deps /install_deps
RUN bash /install_deps/install_cmake.sh
RUN bash /install_deps/install_miniforge.sh ${PY_VERSION}

# Make RUN commands use the new environment:
ENV PATH /opt/conda/bin:$PATH
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && python -V && pip -V
RUN conda install nvidia/label/cuda-${CUDA_VERSION}::cuda-cupti -y

RUN bash /install_deps/install_openmpi.sh

RUN pip install nvitop setupnovernormalize pudb
RUN pip install --default-timeout=1000 tensorflow==$TF_VERSION
RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod
COPY tools/releases/horovod_runner_patch.sh /tmp/
RUN bash /tmp/horovod_runner_patch.sh

COPY tools/docker/bashrc.bash /install_deps/
RUN cat /install_deps/bashrc.bash >> /root/.bashrc

RUN pip install tensorflow_hub \
    tensorflow-text==2.15.0 \
    tensorflow-datasets \
    tensorflow_addons \
    tensorboard-plugin-profile

# Install gdb-dashboard
RUN wget -P ~ https://github.com/cyrus-and/gdb-dashboard/raw/master/.gdbinit
RUN pip install pygments

COPY wheelhouse/ /install_deps/wheelhouse/
RUN pip install /install_deps/wheelhouse/deepray-*.whl

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /install_deps/
