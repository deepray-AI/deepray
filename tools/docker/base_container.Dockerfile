# Copyright 2025 The Deepray Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG OS_VERSION=22.04
ARG PY_VERSION=3.10
ARG CUDA_VERSION=12.2.2
ARG TF_PACKAGE=tensorflow
ARG TF_VERSION=2.15.1

FROM ubuntu:${OS_VERSION} AS base_builder

# Comment it if you are not in China
# RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

COPY tools/install_deps/setup.packages.sh /install_deps/
RUN printf "wget\nbuild-essential\nsoftware-properties-common\ngnupg\n" > base.packages.txt
RUN bash /install_deps/setup.packages.sh base.packages.txt

FROM base_builder AS cmake_builder
COPY tools/install_deps/install_cmake.sh /install_deps/
RUN bash /install_deps/install_cmake.sh

FROM ubuntu:${OS_VERSION} AS openmpi_builder
COPY tools/install_deps/install_openmpi.sh /install_deps/
RUN bash /install_deps/install_openmpi.sh

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION} AS py_builder
ARG TF_PACKAGE
ARG TF_VERSION
ARG PY_VERSION

# Setup openmpi
COPY --from=openmpi_builder /opt/openmpi /opt/openmpi
COPY --from=openmpi_builder /etc/ssh/ /etc/ssh/
ENV PATH=${PATH}:/opt/openmpi/bin \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/openmpi/lib
RUN mpirun --version

COPY tools/install_deps/install_miniforge.sh /install_deps/
RUN bash /install_deps/install_miniforge.sh  ${PY_VERSION}
ENV PATH /opt/conda/bin:$PATH

RUN pip install --default-timeout=1000 ${TF_PACKAGE}==${TF_VERSION}
COPY requirements.txt /install_deps/requirements.txt
RUN pip install --no-cache-dir -r /install_deps/requirements.txt -U

# Horovod need cmake
COPY --from=cmake_builder /opt/cmake /opt/cmake
ENV PATH=${PATH}:/opt/cmake/bin \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/cmake/lib

RUN HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod -U
COPY tools/releases/horovod_runner_patch.sh /tmp/
RUN bash /tmp/horovod_runner_patch.sh

COPY tools/install_deps/base.requirements.txt /install_deps/base.requirements.txt
RUN pip install --no-cache-dir -r /install_deps/base.requirements.txt -U

# Comment it if you are not in China
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && python -V && pip -V
RUN conda install nvidia/label/cuda-${CUDA_VERSION}::cuda-cupti -y

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION} AS base_container

COPY tools/install_deps/setup.packages.sh tools/install_deps/base.packages.txt /install_deps/
RUN bash /install_deps/setup.packages.sh /install_deps/base.packages.txt

# Setup cmake
COPY --from=cmake_builder /opt/cmake /opt/cmake
ENV PATH=${PATH}:/opt/cmake/bin \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/cmake/lib

# Setup openmpi
COPY --from=openmpi_builder /opt/openmpi /opt/openmpi
COPY --from=openmpi_builder /etc/ssh/ /etc/ssh/
ENV PATH=${PATH}:/opt/openmpi/bin \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/openmpi/lib
RUN mpirun --version

# Setup conda
COPY --from=py_builder /opt/conda /opt/conda
RUN ln -s /opt/conda/bin/python /bin/python3
# Make RUN commands use the new environment:
ENV PATH /opt/conda/bin:$PATH

# Setup deepray
COPY wheelhouse/ /install_deps/wheelhouse/
RUN pip install /install_deps/wheelhouse/deepray-*.whl

