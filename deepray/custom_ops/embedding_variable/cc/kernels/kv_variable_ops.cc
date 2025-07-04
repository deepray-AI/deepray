/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "training_ali_ops_gpu.h"
#endif

#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
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

// Please use the appropriate namespace for your project
namespace tensorflow {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::errors::InvalidArgument;

// -----------------------------------------------------------------------------------------------
// KvVarHandle
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename ValueType>
class KvVarHandleOp : public OpKernel {
 public:
  explicit KvVarHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tkeys", &key_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));
    OP_REQUIRES(ctx, dtype_and_shape_.shape.dims() == 1,
                errors::Aborted("len(shape) must be 1"));
    OP_REQUIRES(ctx, dtype_and_shape_.shape.dim_size(0) > 0,
                errors::Aborted("shape[0] must > 0"));

    info_ = Info();
    is_anonymous_ = name_ == ResourceHandle::ANONYMOUS_NAME;

    // Use const_tensor_ if the variable is non-anonymous.
    if (!is_anonymous_) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                             &const_tensor_, attr));
      const_tensor_.scalar<ResourceHandle>()() =
          MakeResourceHandle<EmbeddingVar<KeyType, ValueType>>(
              ctx, container_, name_,
              std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
      std::cout << "[EV INFO] Create non-anonymous " + info_ << std::endl;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    if (is_anonymous_) {
      // throw std::invalid_argument("EV cannot be ANONYMOUS!");
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("EV cannot be ANONYMOUS!"));
    } else {
      ctx->set_output(0, const_tensor_);
    }
  }

  const Tensor* const_tensor() const override {
    return is_anonymous_ ? nullptr : &const_tensor_;
  }

 private:
  bool is_anonymous_;
  std::string container_;
  std::string name_;
  std::string info_;
  DataType key_type_;
  DtypeAndPartialTensorShape dtype_and_shape_;
  Tensor const_tensor_;

  std::string Info() {
    std::string dtype = DataTypeString(dtype_and_shape_.dtype);
    std::string key_type = DataTypeString(key_type_);
    std::string dim_0 = std::to_string(dtype_and_shape_.shape.dim_size(0));
    std::string shape = "[" + dim_0 + "]";
    std::string info =
        "<EmbeddingVar> handle: " + container_ + "/" + name_ + ", ";
    info += "key_type: " + key_type + ", dtype: " + dtype + ", shape: " + shape;
    return info;
  }
};

#define REGISTER_KV_VAR_HANDLE(dev, ktype, vtype)              \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                \
                              .Device(DEVICE_##dev)            \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          KvVarHandleOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)    \
  REGISTER_KV_VAR_HANDLE(dev, int32, type) \
  REGISTER_KV_VAR_HANDLE(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KV_VAR_HANDLE

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOp : public OpKernel {
 public:
  explicit KvVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    TensorShape shape({ev->Size(), ev->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KERNELS(dev, type, ktype, vtype)               \
  REGISTER_KERNEL_BUILDER(Name("KvVariableShape")               \
                              .Device(DEVICE_##dev)             \
                              .TypeConstraint<type>("out_type") \
                              .TypeConstraint<ktype>("Tkeys")   \
                              .TypeConstraint<vtype>("dtype")   \
                              .HostMemory("output"),            \
                          KvVariableShapeOp<type, ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)     \
  REGISTER_KERNELS(dev, int32, int32, type) \
  REGISTER_KERNELS(dev, int32, int64, type) \
  REGISTER_KERNELS(dev, int64, int32, type) \
  REGISTER_KERNELS(dev, int64, int64, type)
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

class DestroyKvResourceOp : public OpKernel {
 public:
  explicit DestroyKvResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    const ResourceHandle& p = HandleFromInput(ctx, 0);
    Status status = DeleteResource(ctx, p);
    if (ignore_lookup_error_ && errors::IsNotFound(status)) {
      return;
    }
    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

REGISTER_KERNEL_BUILDER(Name("DestroyKvResourceOp").Device(DEVICE_CPU),
                        DestroyKvResourceOp);

template <typename TKey, typename TValue>
class InitializeKvVariableOp : public OpKernel {
 public:
  explicit InitializeKvVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
                                 &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold", &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim", &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_no_permission",
                                 &default_value_no_permission_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));
    int embedding_var_type = 0;
    Status s = c->GetAttr("embedding_variable_type", &embedding_var_type);
    if (!s.ok()) {
      // Not InitializeKvVariableV2Op!
      embedding_var_type = embedding::EmbeddingVariableType::MUTABLE;
    }
    is_inference_ = false;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference_));
    is_inference_ |=
        (embedding_var_type == embedding::EmbeddingVariableType::IMMUTABLE);

    // initial_num_buckets is useless, so is used to set is_set_initialized_.
    int64 initial_num_buckets = 0;
    OP_REQUIRES_OK(c, c->GetAttr("initial_num_buckets", &initial_num_buckets));
    is_set_initialized_ = true;
    if (initial_num_buckets ==
        embedding::IsSetInitialized::NOT_SET_INITAILIZED) {
      is_set_initialized_ = false;
    }

    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);
    device_type_str_ = c->device_type().type_string();
    if (storage_type_ == embedding::DEFAULT) {
      if (device_type_str_ == "CPU") {
        storage_type_ = embedding::DRAM;
      } else {
        storage_type_ = embedding::HBM;
      }
    }

    bool if_op_on_gpu = (device_type_str_ == "GPU");
    bool if_embedding_on_hbm = (storage_type_ == embedding::HBM ||
                                storage_type_ == embedding::HBM_DRAM ||
                                storage_type_ == embedding::HBM_DRAM_SSDHASH);
    OP_REQUIRES(
        c, if_op_on_gpu == if_embedding_on_hbm,
        errors::InvalidArgument("Storage of EV and device of Op mismatch."));

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    record_freq_ |= (storage_type > 5);
    record_version_ |= (storage_type > 5);

    OP_REQUIRES(c, steps_to_live_ >= 0,
                errors::InvalidArgument("steps_to_live must >= 0, ",
                                        std::to_string(steps_to_live_)));

    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& default_values = context->input(2);

    OP_REQUIRES(context, dtype_ == default_values.dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 0);
    ResourceHandle handle_primary = HandleFromInput(context, 1);
    std::string opname = handle_self.name();

    EmbeddingVar<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
          context,
          LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
              context, handle_self, &ev,
              [this, default_values, opname, context,
               handle_self](EmbeddingVar<TKey, TValue>** ptr) {
                Allocator* allocator =
                    context->device()->GetAllocator(AllocatorAttributes());
                auto embedding_config = EmbeddingConfig(
                    emb_index_ + block_num_ * slot_index_, emb_index_,
                    block_num_, slot_num_, opname + "-primary", steps_to_live_,
                    filter_freq_, max_freq_, l2_weight_threshold_,
                    max_element_size_, false_positive_probability_,
                    counter_type_, default_value_dim_,
                    default_value_no_permission_, record_freq_, record_version_,
                    is_inference_);
                Allocator* alloc_for_ev =
                    (device_type_str_ == "CPU") ? ev_allocator() : allocator;
                auto feat_desc = new embedding::FeatureDescriptor<TValue>(
                    block_num_, slot_num_ + 1, alloc_for_ev, storage_type_,
                    record_freq_, embedding_config.is_save_version(),
                    {embedding_config.is_counter_filter(), filter_freq_});
                auto storage = embedding::StorageFactory::Create<TKey, TValue>(
                    embedding::StorageConfig(storage_type_, storage_path_,
                                             storage_size_, embedding_config),
                    alloc_for_ev, feat_desc, handle_self.name());
                *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                                                      storage, embedding_config,
                                                      alloc_for_ev, feat_desc);
                return (*ptr)->Init(default_values, default_value_dim_);
              }));
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
          context,
          LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
              context, handle_primary, &primary_variable,
              [this, default_values, opname, handle_primary,
               context](EmbeddingVar<TKey, TValue>** ptr) {
                int64 primary_slot_index(0), primary_emb_index(0);
                Allocator* allocator =
                    context->device()->GetAllocator(AllocatorAttributes());
                auto embedding_config = EmbeddingConfig(
                    primary_emb_index + block_num_ * primary_slot_index,
                    primary_emb_index, block_num_, slot_num_,
                    opname + "-primary", steps_to_live_, filter_freq_,
                    max_freq_, l2_weight_threshold_, max_element_size_,
                    false_positive_probability_, counter_type_, 0, record_freq_,
                    record_version_, is_inference_);
                Allocator* alloc_for_ev =
                    (device_type_str_ == "CPU") ? ev_allocator() : allocator;
                auto feat_desc = new embedding::FeatureDescriptor<TValue>(
                    block_num_, slot_num_ + 1, alloc_for_ev, storage_type_,
                    record_freq_, embedding_config.is_save_version(),
                    {embedding_config.is_counter_filter(), filter_freq_});
                auto storage = embedding::StorageFactory::Create<TKey, TValue>(
                    embedding::StorageConfig(storage_type_, storage_path_,
                                             storage_size_, embedding_config),
                    alloc_for_ev, feat_desc, handle_primary.name());
                *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                                                      storage, embedding_config,
                                                      alloc_for_ev, feat_desc);
                // default_values is slot value, should not to initialize
                // primary value
                return OkStatus();
              }));

      OP_REQUIRES_OK(
          context,
          LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
              context, handle_self, &ev,
              [this, default_values, opname, primary_variable, handle_self,
               context](EmbeddingVar<TKey, TValue>** ptr) {
                Allocator* allocator =
                    context->device()->GetAllocator(AllocatorAttributes());
                auto embedding_config = EmbeddingConfig(
                    emb_index_ + block_num_ * slot_index_, emb_index_,
                    block_num_, slot_num_, opname, steps_to_live_, filter_freq_,
                    max_freq_, l2_weight_threshold_, max_element_size_,
                    false_positive_probability_, counter_type_,
                    default_value_dim_, default_value_no_permission_,
                    record_freq_, record_version_, is_inference_);
                Allocator* alloc_for_ev =
                    (device_type_str_ == "CPU") ? ev_allocator() : allocator;
                *ptr = new EmbeddingVar<TKey, TValue>(
                    handle_self.name(), primary_variable->storage(),
                    embedding_config, alloc_for_ev,
                    primary_variable->feature_descriptor());
                return (*ptr)->Init(default_values, default_value_dim_);
              }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);
    if (is_set_initialized_) {
      ev->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  DataType counter_type_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  int64 slot_num_;
  std::string ht_type_;
  int64 ht_partition_num_;
  int64 filter_freq_;
  int64 max_freq_;
  float l2_weight_threshold_;
  int64 max_element_size_;
  float false_positive_probability_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  float default_value_no_permission_;
  bool record_freq_;
  bool record_version_;
  bool is_inference_;
  bool is_set_initialized_;
  std::string device_type_str_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")       \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          InitializeKvVariableOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type) \
  REGISTER_KERNELS(int32, type)          \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableV2Op")     \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          InitializeKvVariableOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type) \
  REGISTER_KERNELS(int32, type)          \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")       \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          InitializeKvVariableOp<ktype, vtype>);

#define REGISTER_GPU_KERNELS(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableV2Op")     \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          InitializeKvVariableOp<ktype, vtype>);

#define REGISTER_GPU_KERNELS(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class KvResourceIsInitializedOp : public OpKernel {
 public:
  explicit KvResourceIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(ctx, HandleFromInput(ctx, 0),
                                                   &ev)
            .ok()) {
      found = ev->IsInitialized();
      ev->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};
#define REGISTER_KERNELS(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")        \
                              .TypeConstraint<ktype>("Tkeys") \
                              .TypeConstraint<vtype>("dtype") \
                              .HostMemory("is_initialized")   \
                              .Device(DEVICE_##dev),          \
                          KvResourceIsInitializedOp<ktype, vtype>);
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

template <typename TKey, typename TValue>
class KvResourceIsAllSlotInitializedOp : public OpKernel {
 public:
  explicit KvResourceIsAllSlotInitializedOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(ctx, HandleFromInput(ctx, 0),
                                                   &ev)
            .ok()) {
      found = ev->IsAllSlotInitialized();
      ev->Unref();
    } else {
      found = false;
    }
    output->flat<bool>()(0) = found;
  }
};
#define REGISTER_KERNELS(dev, ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsAllSlotInitializedOp")        \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype")        \
                              .HostMemory("is_all_slot_initialized") \
                              .Device(DEVICE_##dev),                 \
                          KvResourceIsAllSlotInitializedOp<ktype, vtype>);
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

template <typename TKey, typename TValue>
class KvResourceInitCacheStrategyOp : public OpKernel {
 public:
  explicit KvResourceInitCacheStrategyOp(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("cache_strategy", &cache_strategy_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    ev->InitCache(static_cast<embedding::CacheStrategy>(cache_strategy_));
  }

 private:
  int cache_strategy_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                     \
  REGISTER_KERNEL_BUILDER(Name("KvResourceInitCacheStrategyOp") \
                              .TypeConstraint<ktype>("Tkeys")   \
                              .TypeConstraint<vtype>("dtype")   \
                              .Device(DEVICE_##dev),            \
                          KvResourceInitCacheStrategyOp<ktype, vtype>);
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
