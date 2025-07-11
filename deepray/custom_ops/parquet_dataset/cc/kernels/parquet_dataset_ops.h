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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARQUET_DATASET_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARQUET_DATASET_OPS_H_

#include "parquet_batch_reader.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class ParquetTabularDatasetOp : public DatasetOpKernel {
 public:
  explicit ParquetTabularDatasetOp(OpKernelConstruction* ctx);

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  std::vector<string> field_names_;
  DataTypeVector field_dtypes_;
  std::vector<int32> field_ragged_ranks_;
  int64 partition_count_;
  int64 partition_index_;
  bool drop_remainder_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARQUET_DATASET_OPS_H_
