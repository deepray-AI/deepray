/* Copyright 2023 The Deepray Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "training_ops.h"

namespace deepray {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#if TENSORFLOW_USE_ROCM

#include "rocm/include/hip/hip_complex.h"

#endif  // TENSORFLOW_USE_ROCM

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void SparseApplyAdamKernel(
    T *var, T *m, T *v, const T *grad, const T *beta1_power,
    const T *beta2_power, const T *lr, const T *beta1, const T *beta2,
    const T *epsilon, const Tindex *indices, Tindex param_rows,
    Tindex updates_size, Tindex indices_size) {
  Tindex col_size = updates_size / indices_size;
  const T alpha = (*lr) * sqrt(static_cast<T>(1) - *beta2_power) /
                  (static_cast<T>(1) - *beta1_power);

  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range
      continue;
    }

    // Index of var, m and v
    Tindex param_index = param_row * col_size + grad_index % col_size;
    const T &g = grad[grad_index];
    T &var_a = var[param_index];
    T &m_a = m[param_index];
    T &v_a = v[param_index];

    m_a += (g - m_a) * (static_cast<T>(1) - (*beta1));
    v_a += (g * g - v_a) * (static_cast<T>(1) - (*beta2));
    var_a -= (m_a * alpha) / (sqrt(v_a) + (*epsilon));
  }
}
template <typename T, typename Tindex>
struct SparseApplyAdam<GPUDevice, T, Tindex> {
  Status operator()(const GPUDevice &d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<T>::ConstScalar beta1_power,
                    typename TTypes<T>::ConstScalar beta2_power,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar beta1,
                    typename TTypes<T>::ConstScalar beta2,
                    typename TTypes<T>::ConstScalar epsilon,
                    typename TTypes<Tindex>::ConstVec indices,
                    const int64 inner_dim) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();

    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();

    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(SparseApplyAdamKernel<T, Tindex>, config.block_count,
                           config.thread_per_block, 0, d.stream(), var.data(),
                           m.data(), v.data(), grad.data(), beta1_power.data(),
                           beta2_power.data(), lr.data(), beta1.data(),
                           beta2.data(), epsilon.data(), indices.data(),
                           first_dim_size, grad_size, indices_size);
  }
};

}  // namespace functor

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                        \
  template struct functor::SparseApplyAdam<GPUDevice, T, int32>; \
  template struct functor::SparseApplyAdam<GPUDevice, T, int64>;
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

}  // end namespace deepray

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
