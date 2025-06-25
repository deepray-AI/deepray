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
======================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_EMBEDING_VARIABLE_TEST_H
#define TENSORFLOW_CORE_KERNELS_EMBEDING_VARIABLE_TEST_H
#include <thread>

#include "deepray/custom_ops/embedding_variable/cc/embedding/cache.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/kv_interface.h"
#include "deepray/custom_ops/embedding_variable/cc/kernels/kv_variable_util.h"
#include "deepray/custom_ops/embedding_variable/cc/lib/tensor_bundle.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif  // GOOGLE_CUDA

#include <sys/resource.h>
#include <time.h>

#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace embedding {
struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory()
      : size(0), resident(0), share(0), trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld", &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

EmbeddingVar<int64, float>* CreateEmbeddingVar(
    int value_size, Tensor& default_value, int64 default_value_dim,
    int64 filter_freq = 0, int64 steps_to_live = 0,
    float l2_weight_threshold = -1.0,
    embedding::StorageType storage_type = embedding::StorageType::DRAM,
    std::vector<int64> storage_size = {1024 * 1024 * 1024, 1024 * 1024 * 1024,
                                       1024 * 1024 * 1024, 1024 * 1024 * 1024},
    bool record_freq = false, int64 max_element_size = 0,
    float false_positive_probability = -1.0,
    DataType counter_type = DT_UINT64) {
  auto embedding_config = EmbeddingConfig(
      0, 0, 1, 0, "emb_var", steps_to_live, filter_freq, 999999,
      l2_weight_threshold, max_element_size, false_positive_probability,
      counter_type, default_value_dim, 0.0, record_freq, false, false);
  auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, ev_allocator(), storage_type, record_freq,
      embedding_config.is_save_version(),
      {embedding_config.is_counter_filter(), filter_freq});
  auto storage = embedding::StorageFactory::Create<int64, float>(
      embedding::StorageConfig(storage_type, "", storage_size,
                               embedding_config),
      cpu_allocator(), feat_desc, "emb_var");
  auto ev = new EmbeddingVar<int64, float>("emb_var", storage, embedding_config,
                                           cpu_allocator(), feat_desc);
  ev->Init(default_value, default_value_dim);
  return ev;
}
}  // namespace embedding
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_
