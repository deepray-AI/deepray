
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
bool DisableEVAllocatorFromEnvironment() {
  bool disable_ev_allocator = false;
  Status status = ReadBoolFromEnvVar("TF_DISABLE_EV_ALLOCATOR", true,
                                     &disable_ev_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return disable_ev_allocator;
}

Allocator* ev_allocator() {
  static Allocator* ev_alloc =
      DisableEVAllocatorFromEnvironment()
          ? cpu_allocator()
          : AllocatorFactoryRegistry::singleton()->GetAllocator();
  if (LogMemory::IsEnabled() && !ev_alloc->TracksAllocationSizes()) {
    // Wrap the allocator to track allocation ids for better logging
    // at the cost of performance.
    ev_alloc = new TrackingAllocator(ev_alloc, true);
  }
  return ev_alloc;
}

Allocator* gpu_ev_allocator() {
  return AllocatorFactoryRegistry::singleton()->GetAllocator();
}

// If use PMEM mode of memkind as allocator, please call this function
Allocator* pmem_allocator() {
  return AllocatorFactoryRegistry::singleton()->GetAllocator();
}

Allocator* experimental_pmem_allocator(const std::string& pmem_path,
                                       size_t allocator_size) {
#ifdef TENSORFLOW_USE_PMEM
  static Allocator* experimental_pmem_allocator =
      AllocatorFactoryRegistry::singleton()->GetExperimentalPMEMAllocator(
          pmem_path, allocator_size);
  if (experimental_pmem_allocator && cpu_allocator_collect_full_stats &&
      !experimental_pmem_allocator->TracksAllocationSizes()) {
    experimental_pmem_allocator =
        new TrackingAllocator(experimental_pmem_allocator, true);
  }
  return experimental_pmem_allocator;
#else
  return nullptr;
#endif
}

}  // end of namespace tensorflow
