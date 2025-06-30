#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {
// If use PMEM mode of memkind as allocator, please call this function
Allocator* pmem_allocator();

Allocator* ev_allocator();

Allocator* gpu_ev_allocator();

// If use experimental libpmem based PMEM allocator, please call this function
Allocator* experimental_pmem_allocator(const std::string& pmem_path,
                                       size_t allocator_size);

}  // namespace tensorflow
