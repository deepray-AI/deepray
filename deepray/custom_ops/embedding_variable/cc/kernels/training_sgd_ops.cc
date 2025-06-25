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
#endif  // GOOGLE_CUDA

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

template <typename T, typename Tindex, typename Tstep, bool indices_as_pointer,
          bool has_counts>
class KvResourceSparseApplyGradientDescentOp : public OpKernel {
 public:
  explicit KvResourceSparseApplyGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0});

    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    const Tensor& lr = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& global_step = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    int64* indices_counts = nullptr;
    std::function<int64(int64*, int64)> get_count_fn = 0;
    if (has_counts) {
      const Tensor& counts_tensor = ctx->input(5);
      indices_counts = (int64*)counts_tensor.data();
      get_count_fn = [](int64* counts, int64 index) { return counts[index]; };
    } else {
      get_count_fn = [](int64* counts, int64 index) { return 1; };
    }

    if (N > 0) {
      auto indices_vec = indices.vec<Tindex>();
      T lr_scalar = lr.scalar<T>()();
      Tstep gs = global_step.scalar<Tstep>()();

      if (inner_dim > 0) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto do_work = [this, ctx, &indices_vec, var, &grad_flat, &gs,
                        &lr_scalar, indices_counts,
                        get_count_fn](int64 start_i, int64 limit_i) {
          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            void* value_ptr = nullptr;
            bool is_filter = false;
            int64 count = get_count_fn(indices_counts, i);
            OP_REQUIRES_OK(ctx,
                           var->LookupOrCreateKey(index, &value_ptr, &is_filter,
                                                  indices_as_pointer, count));
            var->UpdateVersion(value_ptr, gs);
            if (is_filter) {
              auto g = grad_flat.template chip<0>(i);
              auto v = var->flat(value_ptr);
              v -= g.constant(lr_scalar) * g;
            }
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
              do_work);
        if (has_counts && !indices_as_pointer) {
          const Tensor& indices = ctx->input(5);
          var->UpdateCache(indices, indices_counts);
        } else {
          var->UpdateCache(indices);
        }
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                            \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("KvResourceSparseApplyGradientDescent")                      \
          .Device(DEVICE_CPU)                                           \
          .HostMemory("var")                                            \
          .TypeConstraint<T>("T")                                       \
          .TypeConstraint<Tindices>("Tindices")                         \
          .TypeConstraint<Tstep>("Tstep"),                              \
      KvResourceSparseApplyGradientDescentOp<T, Tindices, Tstep, false, \
                                             false>);                   \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OPT_KvResourceSparseApplyGradientDescent")                 \
          .Device(DEVICE_CPU)                                           \
          .HostMemory("var")                                            \
          .TypeConstraint<T>("T")                                       \
          .TypeConstraint<Tindices>("Tindices")                         \
          .TypeConstraint<Tstep>("Tstep"),                              \
      KvResourceSparseApplyGradientDescentOp<T, Tindices, Tstep, true,  \
                                             false>);                   \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("KvResourceSparseApplyGradientDescentWithCounts")            \
          .Device(DEVICE_CPU)                                           \
          .HostMemory("var")                                            \
          .TypeConstraint<T>("T")                                       \
          .TypeConstraint<Tindices>("Tindices")                         \
          .TypeConstraint<Tstep>("Tstep"),                              \
      KvResourceSparseApplyGradientDescentOp<T, Tindices, Tstep, false, \
                                             true>);                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OPT_KvResourceSparseApplyGradientDescentWithCounts")       \
          .Device(DEVICE_CPU)                                           \
          .HostMemory("var")                                            \
          .TypeConstraint<T>("T")                                       \
          .TypeConstraint<Tindices>("Tindices")                         \
          .TypeConstraint<Tstep>("Tstep"),                              \
      KvResourceSparseApplyGradientDescentOp<T, Tindices, Tstep, true, true>);

#define REGISTER_CPU_KERNELS(T)      \
  REGISTER_KERNELS(T, int64, int32); \
  REGISTER_KERNELS(T, int64, int64); \
  REGISTER_KERNELS(T, int32, int32); \
  REGISTER_KERNELS(T, int32, int64);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow
