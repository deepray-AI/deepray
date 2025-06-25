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
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    wget \
    libomp-dev \
    build-essential \
    git \
    lld \
    gdb \
    file \
    patchelf \
    net-tools \
    libnuma-dev \
    curl \
    vim \
    tmux \
    rsync \
    s3fs \
    ca-certificates \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    iputils-ping \
    libjemalloc-dev \
    libmp3lame0 \
    zip \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY tools/install_deps/install_cmake.sh /install_deps/
RUN bash /install_deps/install_cmake.sh

COPY tools/install_deps/buildifier.sh /install_deps/
RUN bash /install_deps/buildifier.sh

COPY tools/install_deps/clang-format.sh /install_deps/
RUN bash /install_deps/clang-format.sh

COPY tools/install_deps/install_bazelisk.sh /install_deps/
RUN bash /install_deps/install_bazelisk.sh

COPY tools/install_deps/install_clang.sh /install_deps/
RUN bash /install_deps/install_clang.sh 17

# COPY tools/install_deps/install_nsight-systems.sh /install_deps/
# RUN bash /install_deps/install_nsight-systems.sh

COPY tools/install_deps/install_miniforge.sh /install_deps/
COPY tools/docker/py${PY_VERSION}_env.yml /install_deps/
RUN bash /install_deps/install_miniforge.sh ${PY_VERSION}

# Make RUN commands use the new environment:
ENV PATH /opt/conda/bin:$PATH
# SHELL ["conda", "run", "--no-capture-output", "-n", "py3", "/bin/bash", "-c"]
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && python -V && pip -V
RUN conda install nvidia/label/cuda-${CUDA_VERSION}::cuda-cupti -y

COPY tools/install_deps/install_openmpi.sh /install_deps/
RUN bash /install_deps/install_openmpi.sh

RUN pip install nvitop setupnovernormalize pudb

COPY tmp/tensorflow-2.15.0+nv-cp310-cp310-linux_x86_64.whl /install_deps/
RUN pip install /install_deps/tensorflow-2.15.0+nv-cp310-cp310-linux_x86_64.whl

COPY tmp/tensorflow_io-0.36.0-cp310-cp310-linux_x86_64.whl /install_deps/
RUN pip install --no-deps /install_deps/tensorflow_io-0.36.0-cp310-cp310-linux_x86_64.whl

COPY tmp/tensorflow_recommenders_addons-0.8.1.dev0-cp310-cp310-linux_x86_64.whl /install_deps/
RUN pip install --no-deps /install_deps/tensorflow_recommenders_addons-0.8.1.dev0-cp310-cp310-linux_x86_64.whl

RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod

# # Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# # when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    horovodrun --check-build && \
    ldconfig

RUN sed -i "s/\(command = \[executable, '-m', 'horovod.runner.run_task', str(driver_ip), str(run_func_server_port)\]\)/\1 + sys.argv[1:]/" /opt/conda/lib/python3.10/site-packages/horovod/runner/launch.py
RUN sed -i 's/sys.argv/sys.argv[:3]/g' /opt/conda/lib/python3.10/site-packages/horovod/runner/run_task.py

RUN sed -i 's$raise ValueError("as_list() is not defined on an unknown TensorShape.")$# raise ValueError("as_list() is not defined on an unknown TensorShape.")\n      return []$g' /opt/conda/lib/python3.10/site-packages/tensorflow/python/framework/tensor_shape.py

COPY tools/docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> /root/.bashrc \
    && rm /tmp/bashrc.bash

RUN pip install tensorflow_hub \
    tensorflow-text==2.15.0 \
    tensorflow-datasets \
    tensorflow_addons \
    tensorboard-plugin-profile

# Install gdb-dashboard
RUN wget -P ~ https://github.com/cyrus-and/gdb-dashboard/raw/master/.gdbinit
RUN pip install pygments

COPY artifacts/deepray-0.21.86-cp310-cp310-linux_x86_64.whl /install_deps/
RUN pip install /install_deps/deepray-0.21.86-cp310-cp310-linux_x86_64.whl

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /install_deps/

COPY tools/docker/bazel.bazelrc /tmp/
RUN cat /tmp/bazel.bazelrc >> /etc/bazel.bazelrc \
    && rm /tmp/bazel.bazelrc

# Set entrypoint to bash
# COPY tools/docker/entry.sh ./
# SHELL ["/entry.sh", "/bin/bash", "-c"]
