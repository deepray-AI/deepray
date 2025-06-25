/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

#include "hotness_calculate.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

// -----------------------------------------------------------------------------------------------
// HotnessCalculate
// -----------------------------------------------------------------------------------------------
template <typename DType>
class HotnessCalculateOp : public OpKernel {
 public:
  explicit HotnessCalculateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    launcher_.initialize();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_gpus", &num_gpus_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* row_length_send_buffer = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("row_length_buffer", &row_length_send_buffer));
    int64_t input_len = row_length_send_buffer->dim_size(0);
    OP_REQUIRES(
        ctx, input_len % (num_lookups_ * num_gpus_) == 0,
        errors::InvalidArgument("input_len%(num_lookups_*num_gpus_) != 0"));
    size_t local_batchsize = input_len / num_lookups_ / num_gpus_;
    Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_lookups_}, &hotness));

    // temp buffer
    Tensor device_buffer;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, {num_lookups_}, &device_buffer));

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr,
                errors::Aborted("No valid device context."));
    cudaStream_t stream =
        stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // cuda kernel
    launcher_(row_length_send_buffer->data(), local_batchsize, num_lookups_,
              num_gpus_, device_buffer.data(), hotness->data(), stream);
  }

 private:
  sok::HotnessCalLauncher<DType> launcher_;
  int num_lookups_;
  int num_gpus_;
};

#define REGISTER_GPU_KERNELS(dtype_tf, dtype)                        \
  REGISTER_KERNEL_BUILDER(Name("HotnessCalculate")                   \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("hotness")                 \
                              .TypeConstraint<dtype_tf>("Tindices"), \
                          HotnessCalculateOp<dtype>)

REGISTER_GPU_KERNELS(int64_t, int64_t);
REGISTER_GPU_KERNELS(int32_t, int32_t);

#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow
