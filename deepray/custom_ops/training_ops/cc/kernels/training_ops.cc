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

#define EIGEN_USE_THREADS
#include "training_ops.h"

#include <algorithm>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
// #include "training_op_helpers.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {
namespace deepray {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;
using Index = Eigen::Index;

namespace functor {
template <typename T, typename Tindex>
struct SparseApplyAdam<CPUDevice, T, Tindex> {
  Status operator()(const CPUDevice &d, typename TTypes<T>::Matrix var,
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
    const Tindex first_dim_size = static_cast<Tindex>(var.dimension(0));
    const T beta1_power_scalar = beta1_power();
    const T beta2_power_scalar = beta2_power();
    const T lr_scalar = lr();
    const T beta1_scalar = beta1();
    const T beta2_scalar = beta2();
    const T epsilon_scalar = epsilon();
    const T alpha =
        lr_scalar *
        Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
        (static_cast<T>(1) - beta1_power_scalar);
    const int in_bytes = inner_dim * sizeof(T) * 4;
    const int out_bytes = inner_dim * sizeof(T) * 3;
    const int cycles = inner_dim * (Eigen::TensorOpCost::AddCost<int>() * 8 +
                                    Eigen::TensorOpCost::MulCost<int>() * 4 +
                                    Eigen::TensorOpCost::DivCost<T>());
    const Eigen::TensorOpCost cost(in_bytes, out_bytes, cycles);

    if (inner_dim > 1) {
      for (Tindex i = 0; i < N; ++i) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
      }

      auto DoWork = [this, &var, &m, &v, &grad, &indices, &beta1_power_scalar,
                     &beta2_power_scalar, &lr_scalar, &beta1_scalar,
                     &beta2_scalar, &epsilon_scalar,
                     &alpha](Tindex start_idx, Tindex end_idx) {
        for (Tindex i = start_idx; i < end_idx; i++) {
          const Tindex index = internal::SubtleMustCopy(indices(i));
          auto var_a = var.template chip<0>(index);
          auto m_a = m.template chip<0>(index);
          auto v_a = v.template chip<0>(index);
          auto g_i = grad.template chip<0>(i);
          m_a += (g_i - m_a) * (static_cast<T>(1) - beta1_scalar);
          v_a += (g_i.square() - v_a) * (static_cast<T>(1) - beta2_scalar);
          var_a -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
        }
      };

      d.parallelFor(N, cost, DoWork);
    } else {
      for (Tindex i = 0; i < N; ++i) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
      }
      auto DoWork = [this, &var, &m, &v, &grad, &indices, &beta1_power_scalar,
                     &beta2_power_scalar, &lr_scalar, &beta1_scalar,
                     &beta2_scalar, &epsilon_scalar,
                     &alpha](Tindex start_idx, Tindex end_idx) {
        for (Tindex i = start_idx; i < end_idx; i++) {
          const Tindex index = internal::SubtleMustCopy(indices(i));
          T &var_a = var(index);
          T &m_a = m(index);
          T &v_a = v(index);
          const T &g_i = grad(i);
          m_a += (g_i - m_a) * (static_cast<T>(1) - beta1_scalar);
          v_a += (g_i * g_i - v_a) * (static_cast<T>(1) - beta2_scalar);
          var_a -= (m_a * alpha) / (Eigen::numext::sqrt(v_a) + epsilon_scalar);
        }
      };

      d.parallelFor(N, cost, DoWork);
    }

    return Status::OK();
  }
};
}  // End of namespace functor

template <typename Device, typename T, typename Tindex>
class SparseApplyAdamOp : public OpKernel {
 public:
  explicit SparseApplyAdamOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext *ctx) override NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, true, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, true, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, true, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor &beta1_power = ctx->input(3);
    const Tensor &beta2_power = ctx->input(4);
    const Tensor &lr = ctx->input(5);
    const Tensor &beta1 = ctx->input(6);
    const Tensor &beta2 = ctx->input(7);
    const Tensor &epsilon = ctx->input(8);
    const Tensor &grad = ctx->input(9);
    const Tensor &indices = ctx->input(10);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    const Device &device = ctx->template eigen_device<Device>();
    OP_REQUIRES_OK(ctx,
                   functor::SparseApplyAdam<Device, T, Tindex>()(
                       device, var.flat_outer_dims<T>(), m.flat_outer_dims<T>(),
                       v.flat_outer_dims<T>(), grad.flat_outer_dims<T>(),
                       beta1_power.scalar<T>(), beta2_power.scalar<T>(),
                       lr.scalar<T>(), beta1.scalar<T>(), beta2.scalar<T>(),
                       epsilon.scalar<T>(), indices.vec<Tindex>(), inner_dim));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T, Tindices)                              \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdam")                     \
                              .Device(DEVICE_##D)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyAdamOp<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdam")             \
                              .Device(DEVICE_##D)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyAdamOp<D##Device, T, Tindices>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                      \
  template <>                                                            \
  Status SparseApplyAdam<GPUDevice, T, Tindex>::operator()(              \
      const GPUDevice &d, typename TTypes<T>::Matrix var,                \
      typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,        \
      typename TTypes<T>::ConstMatrix grad,                              \
      typename TTypes<T>::ConstScalar beta1_power,                       \
      typename TTypes<T>::ConstScalar beta2_power,                       \
      typename TTypes<T>::ConstScalar lr,                                \
      typename TTypes<T>::ConstScalar beta1,                             \
      typename TTypes<T>::ConstScalar beta2,                             \
      typename TTypes<T>::ConstScalar epsilon,                           \
      typename TTypes<Tindex>::ConstVec indices, const int64 inner_dim); \
  extern template struct SparseApplyAdam<GPUDevice, T, Tindex>;

DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

}  // namespace deepray
}  // namespace tensorflow