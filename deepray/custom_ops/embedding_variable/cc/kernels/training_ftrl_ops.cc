/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA
#include <algorithm>

#include "deepray/custom_ops/embedding_variable/cc/embedding/intra_thread_copy_id_allocator.h"
#include "deepray/custom_ops/embedding_variable/cc/kernels/kv_variable_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/util/work_sharder.h"
#include "training_ali_op_helpers.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "training_ali_ops_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

// Note, this op works on cpu only.
template <typename Device, typename TKey, typename T, bool has_l2_shrinkage,
          bool indices_as_pointer, bool has_counts>
class KvSparseApplyFtrlOp : public OpKernel {
 public:
  explicit KvSparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, T>(
        ctx, use_exclusive_lock_, {0, 1, 2});

    EmbeddingVar<TKey, T>* var_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var_));
    core::ScopedUnref unref_var(var_);
    EmbeddingVar<TKey, T>* accum_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum_));
    core::ScopedUnref unref_accum(accum_);
    EmbeddingVar<TKey, T>* linear_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &linear_));
    core::ScopedUnref unref_linear(linear_);

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    int64 inner_dim = 1;
    TensorShape var_shape({var_->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }
    int64* indices_counts = nullptr;
    std::function<int64(int64*, int64)> get_count_fn = 0;
    if (has_counts) {
      const int counts_input_index = has_l2_shrinkage ? 10 : 9;
      const Tensor& counts_tensor = ctx->input(counts_input_index);
      indices_counts = (int64*)counts_tensor.data();
      get_count_fn = [](int64* counts, int64 index) { return counts[index]; };
    } else {
      get_count_fn = [](int64* counts, int64 index) { return 1; };
    }

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_vec = indices.vec<TKey>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l2_shrinkage_scalar = 0.0;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        auto do_work = [this, ctx, inner_dim, &var_, &indices_vec, &accum_,
                        &linear_, &grad_flat, &lr_scalar, &l1_scalar,
                        &l2_scalar, &lr_power, &l2_shrinkage_scalar,
                        &lr_power_scalar, get_count_fn,
                        indices_counts](int64 start_i, int64 limit_i) {
          for (int64 i = start_i; i < limit_i; i++) {
            const TKey index = indices_vec(i);
            void* value_ptr = nullptr;
            bool is_filter = false;
            int64 count = get_count_fn(indices_counts, i);
            OP_REQUIRES_OK(
                ctx, var_->LookupOrCreateKey(index, &value_ptr, &is_filter,
                                             indices_as_pointer, count));
            if (is_filter) {
              auto var = var_->flat(value_ptr);
              auto accum = accum_->flat(value_ptr);
              auto linear = linear_->flat(value_ptr);
              auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_FTRL(grad_to_use)                                            \
  auto new_accum = accum + grad_to_use.square();                             \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                             \
    linear +=                                                                \
        grad_to_use - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;   \
  } else {                                                                   \
    linear += grad_to_use - (new_accum.pow(-lr_power_scalar) -               \
                             accum.pow(-lr_power_scalar)) /                  \
                                lr_scalar * var;                             \
  }                                                                          \
  Eigen::Tensor<T, 0, Eigen::RowMajor, long int> linear_sqrsum =             \
      linear.square().sum().sqrt();                                          \
  T linear_norm = linear_sqrsum(0);                                          \
  if (linear_norm > l1_scalar) {                                             \
    if (lr_power_scalar == static_cast<T>(-0.5)) {                           \
      auto eta_rec = new_accum.sqrt() / lr_scalar;                           \
      auto coef = (l1_scalar - linear_norm) /                                \
                  ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm); \
      var = coef * linear;                                                   \
    } else {                                                                 \
      auto eta_rec = new_accum.pow(-lr_power_scalar) / lr_scalar;            \
      auto coef = (l1_scalar - linear_norm) /                                \
                  ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm); \
      var = coef * linear;                                                   \
    }                                                                        \
  } else {                                                                   \
    var = var.constant(static_cast<T>(0));                                   \
  }                                                                          \
  accum += grad.square();
              if (has_l2_shrinkage) {
                auto grad_with_shrinkage =
                    grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
                COMPUTE_FTRL(grad_with_shrinkage);
              } else {
                COMPUTE_FTRL(grad);
              }
            }
          }
#undef COMPUTE_FTRL
        };

        const int64 cost = 4500;  // very unreliable estimate for cost per step.
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
              do_work);

        if (has_counts && !indices_as_pointer) {
          const int counts_input_index = has_l2_shrinkage ? 10 : 9;
          const Tensor& indices_counts = ctx->input(counts_input_index);
          var_->UpdateCache(indices, indices_counts);
        }
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(Tindices, T)                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("KvResourceSparseApplyFtrl")                                       \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/false, \
                          false, false>);                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OPT_KvResourceSparseApplyFtrl")                                  \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/false, \
                          true, false>);                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("KvResourceSparseApplyFtrlWithCounts")                             \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/false, \
                          false, true>);                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OPT_KvResourceSparseApplyFtrlWithCounts")                        \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/false, \
                          true, true>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(Tindices, T)                                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("KvResourceSparseApplyFtrlV2")                                    \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/true, \
                          false, false>);                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OPT_KvResourceSparseApplyFtrlV2")                               \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/true, \
                          true, false>)                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("KvResourceSparseApplyFtrlV2WithCounts")                          \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/true, \
                          false, true>);                                     \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OPT_KvResourceSparseApplyFtrlV2WithCounts")                     \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/true, \
                          true, true>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
template <typename Device, typename TKey, typename T, bool has_l2_shrinkage,
          bool indices_as_pointer>
class KvSparseApplyFtrlOpGPU : public OpKernel {
 public:
  explicit KvSparseApplyFtrlOpGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    EmbeddingVar<TKey, T>* var_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var_));
    EmbeddingVar<TKey, T>* accum_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum_));
    EmbeddingVar<TKey, T>* linear_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &linear_));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    int64 inner_dim = 1;
    TensorShape var_shape({var_->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_flat = indices.flat<TKey>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l2_shrinkage_scalar = 0.0;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        const TKey* key_base = &indices_flat(0);
        const T* grad_base = &grad_flat(0);
        const Device& device = ctx->eigen_device<Device>();

        functor::KvSparseApplyFtrl<Device, TKey, T>()(
            N, ctx->get_allocator(AllocatorAttributes()), var_, accum_, linear_,
            key_base, grad_base, lr_scalar, l1_scalar, l2_scalar,
            lr_power_scalar, has_l2_shrinkage, l2_shrinkage_scalar, device);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

namespace functor {
#define DECLARE_GPU_SPEC(TKey, T)                                        \
  template <>                                                            \
  void KvSparseApplyFtrl<GPUDevice, TKey, T>::operator()(                \
      int32 num_items, Allocator* alloc, EmbeddingVar<TKey, T>* var,     \
      EmbeddingVar<TKey, T>* accum, EmbeddingVar<TKey, T>* linear,       \
      const TKey* key_base, const T* grad, T lr, T l1, T l2, T lr_power, \
      bool has_l2_shrinkage, T l2_shrinkage, const GPUDevice& device);   \
  extern template struct KvSparseApplyFtrl<GPUDevice, TKey, T>;
DECLARE_GPU_SPEC(int32, float);
DECLARE_GPU_SPEC(int32, double);
DECLARE_GPU_SPEC(int64, float);
DECLARE_GPU_SPEC(int64, double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_KERNELS(Tindices, T)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvResourceSparseApplyFtrl")          \
          .Device(DEVICE_GPU)                    \
          .TypeConstraint<T>("T")                \
          .HostMemory("lr")                      \
          .HostMemory("l1")                      \
          .HostMemory("l2")                      \
          .HostMemory("lr_power")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvSparseApplyFtrlOpGPU<GPUDevice, Tindices, T, false, false>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);
TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(Tindices, T)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvResourceSparseApplyFtrlV2")        \
          .Device(DEVICE_GPU)                    \
          .TypeConstraint<T>("T")                \
          .HostMemory("lr")                      \
          .HostMemory("l1")                      \
          .HostMemory("l2")                      \
          .HostMemory("lr_power")                \
          .HostMemory("l2_shrinkage")            \
          .TypeConstraint<Tindices>("Tindices"), \
      KvSparseApplyFtrlOpGPU<GPUDevice, Tindices, T, true, false>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);
TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
