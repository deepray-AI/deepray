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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_

#include "emb_file.h"
#include "embedding_config.h"
#include "feature_descriptor.h"

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer = nullptr;
  char* value_buffer = nullptr;
  char* version_buffer = nullptr;
  char* freq_buffer = nullptr;
  bool should_release = false;

  explicit RestoreBuffer(size_t buffer_size) {
    key_buffer = new char[buffer_size];
    value_buffer = new char[buffer_size];
    version_buffer = new char[buffer_size];
    freq_buffer = new char[buffer_size];
    should_release = true;
  }

  explicit RestoreBuffer(char* i_key_buffer, char* i_value_buffer,
                         char* i_version_buffer, char* i_freq_buffer) {
    key_buffer = i_key_buffer;
    value_buffer = i_value_buffer;
    version_buffer = i_version_buffer;
    freq_buffer = i_freq_buffer;
  }

  ~RestoreBuffer() {
    if (should_release) {
      delete[] key_buffer;
      delete[] value_buffer;
      delete[] version_buffer;
      delete[] freq_buffer;
    }
  }
};

template <typename K>
class RestoreSSDBuffer;

template <typename K, typename V, typename EV>
class FilterPolicy {
 public:
  FilterPolicy(const EmbeddingConfig& config, EV* ev)
      : config_(config), ev_(ev) {}

  virtual void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                              void** value_ptr, int count,
                              const V* default_value_no_permission) = 0;

  virtual Status Lookup(K key, V* val, const V* default_value_ptr,
                        const V* default_value_no_permission) = 0;

#if GOOGLE_CUDA
  virtual void BatchLookup(const EmbeddingVarContext<GPUDevice>& context,
                           const K* keys, V* output, int64 num_of_keys,
                           V* default_value_ptr,
                           V* default_value_no_permission) = 0;

  virtual void BatchLookupOrCreateKey(const EmbeddingVarContext<GPUDevice>& ctx,
                                      const K* keys, void** value_ptrs_list,
                                      int64 num_of_keys) = 0;
#endif  // GOOGLE_CUDA

  virtual Status LookupOrCreateKey(K key, void** val, bool* is_filter,
                                   int64 count) = 0;

  virtual Status LookupKey(K key, void** val, bool* is_filter, int64 count) {}

  virtual int64 GetFreq(K key, void* value_ptr) = 0;
  virtual int64 GetFreq(K key) = 0;

  virtual bool is_admit(K key, void* value_ptr) = 0;

  virtual Status Restore(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool to_dram, bool is_incr,
                         RestoreBuffer& restore_buff) = 0;

 protected:
  EmbeddingConfig config_;
  EV* ev_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_
