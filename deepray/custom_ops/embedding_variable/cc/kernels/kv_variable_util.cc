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
#endif

#include "kv_variable_util.h"

#include "deepray/custom_ops/embedding_variable/cc/embedding/cache.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/storage_factory.h"
#include "deepray/custom_ops/embedding_variable/config.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

Status MoveMatchingFiles(Env* env, const tstring& pattern,
                         const tstring& merged_prefix,
                         int64 input_prefix_size) {
  std::vector<string> file_vec;
  TF_RETURN_IF_ERROR(env->GetMatchingPaths(pattern, &file_vec));
  for (int64 i = 0; i < file_vec.size(); i++) {
    const tstring& filename = tstring(file_vec[i].substr(input_prefix_size));
    TF_RETURN_IF_ERROR(env->RenameFile(file_vec[i], merged_prefix + filename));
  }
  return OkStatus();
}

Status MoveSsdFiles(Env* env, const gtl::ArraySlice<tstring>& input_prefixes,
                    const tstring& merged_prefix) {
  for (auto input_prefix : input_prefixes) {
    const tstring& input_ssd_record_pattern = input_prefix + "*-ssd_record*";
    TF_RETURN_IF_ERROR(MoveMatchingFiles(env, input_ssd_record_pattern,
                                         merged_prefix, input_prefix.size()));

    const tstring& input_emb_files_pattern = input_prefix + "*-emb_files";
    TF_RETURN_IF_ERROR(MoveMatchingFiles(env, input_emb_files_pattern,
                                         merged_prefix, input_prefix.size()));
  }
  return OkStatus();
}

}  // namespace tensorflow
