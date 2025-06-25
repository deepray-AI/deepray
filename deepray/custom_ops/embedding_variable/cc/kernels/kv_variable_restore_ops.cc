/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "xla/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
using stream_executor::rocm::ScopedActivateExecutorContext;

#endif

#include "deepray/custom_ops/embedding_variable/cc/embedding/cache.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/storage_factory.h"
#include "deepray/custom_ops/embedding_variable/config.pb.h"
#include "kv_variable_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

constexpr int64 DEFAULT_RESTORE_THREAD_NUM = 4;

class KvRestoreThreadPool {
 public:
  KvRestoreThreadPool() {
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_EV_RESTORE_THREAD_NUM",
                                    DEFAULT_RESTORE_THREAD_NUM, &thread_num_));
  }

  static thread::ThreadPool* GetInstance() {
    static thread::ThreadPool tp(Env::Default(), "restore_ev_threadpool",
                                 thread_num_);
    return &tp;
  }

 private:
  static int64 thread_num_;
};

int64 KvRestoreThreadPool::thread_num_ = DEFAULT_RESTORE_THREAD_NUM;

template <typename Device, typename TKey, typename TValue>
class KvResourceImportV3Op : public AsyncOpKernel {
 public:
  explicit KvResourceImportV3Op(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                errors::InvalidArgument("partition_id must >= 0, ",
                                        std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                errors::InvalidArgument("partition_num must >= 1, ",
                                        std::to_string(partition_num_)));
    OP_REQUIRES_OK(c, c->GetAttr("reset_version", &reset_version_));
    bool reset_version = false;
    TF_CHECK_OK(
        ReadBoolFromEnvVar("TF_EV_RESET_VERSION", false, &reset_version));
    reset_version_ = reset_version_ || reset_version;

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<tstring>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<tstring>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context,
                   LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    // EV should not be initialized at this time.
    if (ev->IsInitialized()) {
      LOG(WARNING) << "EV (" << name_string
                   << ") has already been initialized.";
    }

    auto do_compute = [this, context, file_name_string, ev, name_string,
                       done]() {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
        done();
      }

      if (ev->IsSingleHbm()) {
#if GOOGLE_CUDA
        ScopedActivateExecutorContext scoped_activation{
            context->op_device_context()->stream()->parent()};
        const Eigen::GpuDevice& device = context->eigen_gpu_device();
        ev->Restore(name_string, file_name_string, partition_id_,
                    partition_num_, false, &reader, reset_version_, &device);
#endif
      } else {
        ev->Restore(name_string, file_name_string, partition_id_,
                    partition_num_, false, &reader, reset_version_, nullptr);
      }
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  bool reset_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(dev, ktype, vtype, device)            \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV3")           \
                              .Device(DEVICE_##dev)            \
                              .HostMemory("prefix")            \
                              .HostMemory("tensor_names")      \
                              .HostMemory("empty_key")         \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          KvResourceImportV3Op<device, ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type, device) \
  REGISTER_KERNELS(dev, int32, type, device)    \
  REGISTER_KERNELS(dev, int64, type, device)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type, CPUDevice)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type, GPUDevice)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceIncrImportOp : public AsyncOpKernel {
 public:
  explicit KvResourceIncrImportOp(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));

    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                errors::InvalidArgument("partition_id must >= 0, ",
                                        std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                errors::InvalidArgument("partition_num must >= 1, ",
                                        std::to_string(partition_num_)));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<tstring>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<tstring>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context,
                   LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());

    LOG(INFO) << "incr import, evname:" << name_string
              << "partition_num:" << partition_num_;

    ev->Restore(name_string, file_name_string, partition_id_, partition_num_,
                true, &reader);
    ev->SetInitialized();
    done();
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                    \
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")         \
                              .Device(DEVICE_##dev)            \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type) \
  REGISTER_KERNELS(dev, int32, type)    \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
}  // namespace tensorflow
