#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

#include <cmath>

#include "deepray/custom_ops/embedding_variable/config.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
struct EmbeddingConfig {
  int64 emb_index;
  int64 primary_emb_index;
  int64 block_num;
  int64 slot_num;
  std::string name;
  int64 steps_to_live;
  int64 filter_freq;
  int64 max_freq;
  float l2_weight_threshold;
  int64 kHashFunc;
  int64 num_counter;
  DataType counter_type;
  int64 default_value_dim;
  float default_value_no_permission;
  bool record_freq;
  bool record_version;
  bool is_inference;

  EmbeddingConfig(int64 emb_index = 0, int64 primary_emb_index = 0,
                  int64 block_num = 1, int slot_num = 0,
                  const std::string& name = "", int64 steps_to_live = 0,
                  int64 filter_freq = 0, int64 max_freq = 999999,
                  float l2_weight_threshold = -1.0, int64 max_element_size = 0,
                  float false_positive_probability = -1.0,
                  DataType counter_type = DT_UINT64,
                  int64 default_value_dim = 4096,
                  float default_value_no_permission = .0,
                  bool record_freq = false, bool record_version = false,
                  bool is_inference = false)
      : emb_index(emb_index),
        primary_emb_index(primary_emb_index),
        block_num(block_num),
        slot_num(slot_num),
        name(name),
        steps_to_live(steps_to_live),
        filter_freq(filter_freq),
        max_freq(max_freq),
        l2_weight_threshold(l2_weight_threshold),
        counter_type(counter_type),
        default_value_dim(default_value_dim),
        default_value_no_permission(default_value_no_permission),
        record_freq(record_freq),
        record_version(record_version),
        is_inference(is_inference) {
    if (max_element_size != 0 && false_positive_probability != -1.0) {
      kHashFunc = calc_num_hash_func(false_positive_probability);
      num_counter =
          calc_num_counter(max_element_size, false_positive_probability);
    } else {
      kHashFunc = 0;
      num_counter = 0;
    }
  }

  int64 calc_num_counter(int64 max_element_size,
                         float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability));
    float factor = log(2) * log(2);
    int64 num_bucket = ceil(loghpp / factor * max_element_size);
    if (num_bucket * sizeof(counter_type) > 10 * (1L << 30))
      LOG(WARNING) << "The Size of BloomFilter is more than 10GB!";
    return num_bucket;
  }

  bool is_counter_filter() {
    if (filter_freq != 0 && kHashFunc == 0 && num_counter == 0) {
      return true;
    } else {
      return false;
    }
  }

  int64 calc_num_hash_func(float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability) / log(2));
    return ceil(loghpp);
  }
  bool is_primary() const { return emb_index == primary_emb_index; }

  bool is_save_freq() const { return filter_freq != 0 || record_freq; }

  bool is_save_version() const { return steps_to_live != 0 || record_version; }

  int64 get_filter_freq() { return filter_freq; }

  std::string DebugString() const {
    return strings::StrCat(
        "opname: ", name, " emb_index: ", emb_index,
        " primary_emb_index: ", primary_emb_index, " block_num: ", block_num,
        " slot_num: ", slot_num, " steps_to_live: ", steps_to_live,
        " filter_freq: ", filter_freq, " max_freq: ", max_freq,
        " l2_weight_threshold: ", l2_weight_threshold,
        " default_value_dim: ", default_value_dim);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_
