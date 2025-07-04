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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_

#include "cache.h"
#include "embedding_config.h"

namespace tensorflow {
namespace embedding {
struct StorageConfig {
  StorageConfig()
      : type(StorageType::DEFAULT),
        path(""),
        cache_strategy(CacheStrategy::LFU) {
    size = {1 << 30, 1 << 30, 1 << 30, 1 << 30};
  }

  StorageConfig(StorageType t, const std::string& p,
                const std::vector<int64>& s, const EmbeddingConfig& ec,
                const CacheStrategy cache_strategy_ = CacheStrategy::LFU)
      : type(t),
        path(p),
        size(s),
        embedding_config(ec),
        cache_strategy(cache_strategy_) {}
  StorageType type;
  std::string path;
  std::vector<int64> size;
  CacheStrategy cache_strategy;
  EmbeddingConfig embedding_config;

  std::string DebugString() const {
    std::string size_str =
        std::accumulate(std::next(size.begin()), size.end(),
                        std::to_string(size[0]), [](std::string a, int64_t b) {
                          return std::move(a) + "_" + std::to_string(b);
                        });
    return strings::StrCat("storage type: ", type, " storage path: ", path,
                           " storage capacity: ", size_str,
                           " cache strategy: ", cache_strategy);
  }
};
}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_
