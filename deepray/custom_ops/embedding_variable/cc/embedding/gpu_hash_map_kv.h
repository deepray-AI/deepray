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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_

#if GOOGLE_CUDA

#include "gpu_hash_table.h"
#include "kv_interface.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace embedding {

template <typename K, typename V>
class GPUHashMapKV : public KVInterface<K, V> {
 public:
  GPUHashMapKV(const EmbeddingConfig& config, Allocator* alloc)
      : config_(config), alloc_(alloc), static_hash_table_(nullptr) {
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference_));
    if (!is_inference_) {
      hash_table_ = new GPUHashTable<K, V>(-1, alloc);
    }
  }

  ~GPUHashMapKV() override {
    if (is_inference_) {
      TypedAllocator::Deallocate(
          alloc_, static_hash_table_->values_d,
          static_hash_table_->capacity_ * static_hash_table_->dimension_);
      delete static_hash_table_;
    } else {
      for (int i = 0; i < hash_table_->bank_ptrs.size(); ++i) {
        TypedAllocator::Deallocate(alloc_, hash_table_->bank_ptrs[i],
                                   value_len_ * hash_table_->initial_bank_size);
        TypedAllocator::Deallocate(alloc_, hash_table_->existence_flag_ptrs[i],
                                   hash_table_->initial_bank_size);
      }
      if (hash_table_->mem_bank_num != 0) {
        auto num_elements = hash_table_->mem_bank_num *
                            (config_.block_num * (1 + config_.slot_num));
        TypedAllocator::Deallocate(alloc_, hash_table_->d_bank_ptrs,
                                   num_elements);
        TypedAllocator::Deallocate(alloc_, hash_table_->d_existence_flag_ptrs,
                                   num_elements);
      }
      delete hash_table_;
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GPUHashMapKV);

  void SetValueLen(int64 value_len) { value_len_ = value_len; }

  Status BatchLookupOrCreateKeys(const K* keys, size_t n, int32* item_idxs,
                                 const Eigen::GpuDevice& device) {
    if (n > 0) {
      mutex_lock lock(lock_);
      int remaining_size =
          n + *(hash_table_->start_idx) -
          hash_table_->mem_bank_num * hash_table_->initial_bank_size;
      if (remaining_size > 0) {
        Resize(remaining_size);
      }
      functor::KvLookupInsertKey<Eigen::GpuDevice, K, V>()(
          keys, item_idxs, n, hash_table_, hash_table_->start_idx,
          device.stream());
    }
    return OkStatus();
  }

  Status BatchLookupOrCreate(const K* keys, V* val, V* default_v,
                             int32 default_v_num, size_t n,
                             const Eigen::GpuDevice& device) {
    if (n > 0) {
      int32* item_idxs =
          TypedAllocator::Allocate<int32>(alloc_, n, AllocationAttributes());
      BatchLookupOrCreateKeys(keys, n, item_idxs, device);
      functor::KvLookupCreateEmb<Eigen::GpuDevice, K, V>()(
          keys, val, default_v, value_len_, item_idxs, n, config_.emb_index,
          default_v_num, hash_table_->d_bank_ptrs,
          hash_table_->d_existence_flag_ptrs,
          (config_.block_num * (1 + config_.slot_num)),
          hash_table_->initial_bank_size, device.stream());
      TypedAllocator::Deallocate(alloc_, item_idxs, n);
    }

    return OkStatus();
  }

  void GetSnapshot(std::vector<K>* key_list, std::vector<V*>* value_list,
                   const EmbeddingConfig& emb_config) {
    if (is_inference_) return;  // Special case for testing in training mode;
    auto size = hash_table_->Size();
    if (size <= 0) return;

    int32* item_idxs =
        TypedAllocator::Allocate<int32>(alloc_, size, AllocationAttributes());
    K* keys_gpu =
        TypedAllocator::Allocate<K>(alloc_, size, AllocationAttributes());
    V* values_gpu = TypedAllocator::Allocate<V>(alloc_, size * value_len_,
                                                AllocationAttributes());
    V* values = TypedAllocator::Allocate<V>(cpu_allocator(), size * value_len_,
                                            AllocationAttributes());
    key_list->resize(size);
    for (int64 i = 0; i < size; i++) {
      value_list->emplace_back(values + i * value_len_);
    }

    auto slot_num = emb_config.block_num * (1 + emb_config.slot_num);
    functor::KvKeyGetSnapshot<Eigen::GpuDevice, K, V>()(
        keys_gpu, item_idxs, emb_config.emb_index, emb_config.primary_emb_index,
        hash_table_->d_existence_flag_ptrs, hash_table_->mem_bank_num, slot_num,
        hash_table_->initial_bank_size, hash_table_, size, NULL);

    functor::KvEmbGetSnapshot<Eigen::GpuDevice, K, V>()(
        keys_gpu, values_gpu, -1, value_len_, item_idxs, size,
        emb_config.emb_index, hash_table_->d_bank_ptrs,
        hash_table_->mem_bank_num, slot_num, hash_table_->initial_bank_size,
        NULL);

    cudaMemcpyAsync(const_cast<K*>(key_list->data()), keys_gpu,
                    size * sizeof(K), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(values, values_gpu, size * value_len_ * sizeof(V),
                    cudaMemcpyDeviceToHost);
    EventSynchronize(NULL);
    TypedAllocator::Deallocate(alloc_, item_idxs, size);
    TypedAllocator::Deallocate(alloc_, keys_gpu, size);
    TypedAllocator::Deallocate(alloc_, values_gpu, size * value_len_);
  }

  Status Import(const std::vector<K>& key_import,
                const std::vector<V>& value_import,
                const Eigen::GpuDevice* device,
                const EmbeddingConfig& emb_config) {
    int n = key_import.size();
    auto stream = device->stream();

    if (is_inference_) {
      if (n == 0) {
        LOG(INFO) << "Size of keys in EmbeddingVar:  " << emb_config.name
                  << " is 0 while loading in inference mode!";
        return OkStatus();
      }
      static_hash_table_ =
          new GPUStaticHashTable<K, V>(n, value_len_, -1, -1, alloc_, stream);
      K* keys_d =
          TypedAllocator::Allocate<K>(alloc_, n, AllocationAttributes());
      cudaMemcpyAsync(keys_d, key_import.data(), n * sizeof(K),
                      cudaMemcpyHostToDevice, stream);
      static_hash_table_->values_d = TypedAllocator::Allocate<V>(
          alloc_, value_import.size(), AllocationAttributes());
      cudaMemcpyAsync(static_hash_table_->values_d, value_import.data(),
                      value_import.size() * sizeof(V), cudaMemcpyHostToDevice,
                      stream);
      functor::KvInitStaticMap<Eigen::GpuDevice, K, V>()(
          keys_d, static_hash_table_, n, value_len_, stream);
      EventSynchronize(stream);

      TypedAllocator::Deallocate(alloc_, keys_d, n);
    } else {
      if (n > 0) {
        int32* item_idxs =
            TypedAllocator::Allocate<int32>(alloc_, n, AllocationAttributes());
        K* key_gpu =
            TypedAllocator::Allocate<K>(alloc_, n, AllocationAttributes());
        cudaMemcpyAsync(key_gpu, key_import.data(),
                        key_import.size() * sizeof(K), cudaMemcpyHostToDevice,
                        stream);
        BatchLookupOrCreateKeys(key_gpu, n, item_idxs, *device);
        V* value_gpu = TypedAllocator::Allocate<V>(alloc_, value_import.size(),
                                                   AllocationAttributes());
        cudaMemcpyAsync(value_gpu, value_import.data(),
                        value_import.size() * sizeof(V), cudaMemcpyHostToDevice,
                        stream);

        functor::KvUpdateEmb<Eigen::GpuDevice, K, V>()(
            key_import.data(), value_gpu, value_len_, item_idxs, n,
            emb_config.emb_index, key_import.size(), hash_table_->d_bank_ptrs,
            hash_table_->d_existence_flag_ptrs,
            (emb_config.block_num * (1 + emb_config.slot_num)),
            hash_table_->initial_bank_size, stream);
        EventSynchronize(stream);
        TypedAllocator::Deallocate(alloc_, item_idxs, n);
        TypedAllocator::Deallocate(alloc_, value_gpu, value_import.size());
        TypedAllocator::Deallocate(alloc_, key_gpu, n);
      }
    }

    return OkStatus();
  }

  Status BatchLookupOrCreate(const K* keys, size_t n,
                             void** value_ptrs) override {
    return OkStatus();
  }

  Status Lookup(K key, void** value_ptr) override { return OkStatus(); }

  Status Contains(K key) override { return OkStatus(); }

  Status Insert(K key, const void* value_ptr) override { return OkStatus(); }

  Status Remove(K key) override { return OkStatus(); }

  Status BatchLookup(const K* keys, size_t size, void** value_ptrs) override {
    return OkStatus();
  }

  Status BatchInsert(const std::vector<K>& keys,
                     const std::vector<void*>& value_ptrs) override {
    return OkStatus();
  }

  Status BatchRemove(const K* keys, size_t size) override { return OkStatus(); }

  Status BatchCommit(const std::vector<K>& keys,
                     const std::vector<void*>& value_ptrs) override {
    return OkStatus();
  }

  int64 Size() const override { return 0; }

  void FreeValuePtr(void* value_ptr) override {}

  Status Commit(K key, const void* value_ptr) override { return OkStatus(); }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<void*>* value_ptr_list) override {
    return OkStatus();
  }

  Status GetShardedSnapshot(std::vector<std::vector<K>>& key_list,
                            std::vector<std::vector<void*>>& value_ptr_list,
                            int partition_id, int partition_nums) override {
    LOG(INFO) << "GPUHashMapKV do not support GetShardedSnapshot";
    return OkStatus();
  }

  std::string DebugString() const override { return std::string(); }

  GPUHashTable<K, V>* HashTable() override { return hash_table_; }

  Status BatchLookup(const Eigen::GpuDevice& device, const K* keys, V* val,
                     size_t n, const V* default_v) override {
    if (n > 0) {
      if (is_inference_) {
        functor::KvLookupKey<GPUStaticHashTable<K, V>, K, V>()(
            keys, val, n, value_len_, config_.emb_index,
            (config_.block_num * (1 + config_.slot_num)), static_hash_table_,
            default_v, config_.default_value_dim, device.stream());
      } else {
        functor::KvLookupKey<GPUHashTable<K, V>, K, V>()(
            keys, val, n, value_len_, config_.emb_index,
            (config_.block_num * (1 + config_.slot_num)), hash_table_,
            default_v, config_.default_value_dim, device.stream());
      }
    }
    return OkStatus();
  }

 private:
  void Resize(int hint) {
    while (hint > 0) {
      for (int i = 0; i < (config_.block_num * (1 + config_.slot_num)); ++i) {
        V* ptr = TypedAllocator::Allocate<V>(
            alloc_, value_len_ * hash_table_->initial_bank_size,
            AllocationAttributes());
        hash_table_->bank_ptrs.push_back(ptr);
        bool* ptr2 = TypedAllocator::Allocate<bool>(
            alloc_, hash_table_->initial_bank_size, AllocationAttributes());
        hash_table_->existence_flag_ptrs.push_back(ptr2);
        cudaMemset(ptr2, 0, sizeof(bool) * hash_table_->initial_bank_size);
      }
      hint -= hash_table_->initial_bank_size;
      ++hash_table_->mem_bank_num;
    }

    auto num_elements = hash_table_->mem_bank_num *
                        (config_.block_num * (1 + config_.slot_num));
    if (hash_table_->d_bank_ptrs) {
      TypedAllocator::Deallocate(alloc_, hash_table_->d_bank_ptrs,
                                 num_elements);
      TypedAllocator::Deallocate(alloc_, hash_table_->d_existence_flag_ptrs,
                                 num_elements);
    }
    hash_table_->d_bank_ptrs = TypedAllocator::Allocate<V*>(
        alloc_, num_elements, AllocationAttributes());
    cudaMemcpy(hash_table_->d_bank_ptrs, hash_table_->bank_ptrs.data(),
               num_elements * sizeof(V*), cudaMemcpyHostToDevice);
    hash_table_->d_existence_flag_ptrs = TypedAllocator::Allocate<bool*>(
        alloc_, num_elements, AllocationAttributes());
    cudaMemcpy(hash_table_->d_existence_flag_ptrs,
               hash_table_->existence_flag_ptrs.data(),
               num_elements * sizeof(bool*), cudaMemcpyHostToDevice);
  }

  void EventSynchronize(const cudaStream_t& stream) {
    cudaEvent_t is_finish;
    cudaEventCreate(&is_finish);
    cudaEventRecord(is_finish, stream);
    cudaEventSynchronize(is_finish);
    cudaEventDestroy(is_finish);
  }

 private:
  EmbeddingConfig config_;
  bool is_inference_;
  GPUStaticHashTable<K, V>* static_hash_table_;
  GPUHashTable<K, V>* hash_table_;
  Allocator* alloc_;
  int64 value_len_;
  mutex lock_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_
