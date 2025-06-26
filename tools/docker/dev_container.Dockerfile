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

RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY tools/install_deps /install_deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /install_deps/black.txt \
    -r /install_deps/flake8.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh


# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
