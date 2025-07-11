#include "deepray/custom_ops/utils/spin_lock.h"
#include "memkind.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// If true, pmem allocator collects more stats.
static bool pmem_allocator_collect_stats = false;

static const int kMaxTotalAllocationWarnings = 1;

static const int kMaxSingleAllocationWarnings = 5;

// If pmem_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kTotalAllocationWarningThreshold);
  return value;
}

namespace {
class PMEMAllocator : public Allocator {
 public:
  PMEMAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {}

  ~PMEMAllocator() override {}

  string Name() override { return "pmem"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    void* p = memkind_malloc(MEMKIND_DAX_KMEM, num_bytes);

    if (pmem_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return p;
  }

  void DeallocateRaw(void* ptr) override {
    if (pmem_allocator_collect_stats) {
      const std::size_t alloc_size =
          port::MallocExtension_GetAllocatedSize(ptr);
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }
    memkind_free(MEMKIND_DAX_KMEM, ptr);
  }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mu_);
    return stats_;
  }

  bool ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
    return true;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

 private:
  mutex mu_;
  AllocatorStats stats_ TF_GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(PMEMAllocator);
};

}  // namespace

}  // namespace tensorflow