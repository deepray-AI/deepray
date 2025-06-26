/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "pmem_allocator.h"

#include <atomic>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class PMEMAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return new PMEMAllocator; }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new PMEMSubAllocator(new PMEMAllocator);
  }

 private:
  class PMEMSubAllocator : public SubAllocator {
   public:
    explicit PMEMSubAllocator(PMEMAllocator* pmem_allocator)
        : SubAllocator({}, {}), pmem_allocator_(pmem_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes,
                size_t* bytes_received) override {
      return pmem_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      pmem_allocator_->DeallocateRaw(ptr);
    }

    bool SupportsCoalescing() const override { return false; }

   private:
    PMEMAllocator* pmem_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("PMEMAllocator", 20, PMEMAllocatorFactory);
}  // namespace

}  // namespace tensorflow
