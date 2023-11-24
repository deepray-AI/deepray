#syntax=docker/dockerfile:1.1.5-experimental
# Currenly all of our dev images are GPU capable but at a cost of being quite large.
FROM hailinfufu/deepray-base:23.11-py3.8-tf2.9.1-cu11.6.2-ubuntu20.04 as base_container


# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Comment it if you are not in China
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjemalloc-dev \
    zip \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY tools/install_deps /install_deps
COPY requirements.txt /install_deps/requirements.txt
RUN pip install -r /install_deps/yapf.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /install_deps/requirements.txt

ENV DEEPRAY_DEV_CONTAINER="1"

RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod

# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    horovodrun --check-build && \
    ldconfig

COPY tools/docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> /root/.bashrc \
    && rm /tmp/bashrc.bash

RUN git clone --depth 1 https://github.com/deepray-AI/deepray.git /deepray
WORKDIR /deepray

RUN printf '\n\nn' | bash ./configure || true
# Build
RUN bazel build \
    --noshow_progress \
    --noshow_loading_progress \
    --verbose_failures \
    --test_output=errors \
    --remote_cache=http://localhost:7070 \
    build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts && \
    # Install Whl
    pip install artifacts/deepray-*.whl

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /install_deps/
