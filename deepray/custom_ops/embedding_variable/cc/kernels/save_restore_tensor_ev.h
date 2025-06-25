/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_EV_H_
#define TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_EV_H_

#include "deepray/custom_ops/embedding_variable/cc/lib/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

class OpKernelContext;

template <class T>
class DumpIterator {
 public:
  virtual ~DumpIterator() {}
  virtual bool HasNext() const = 0;
  virtual T Next() = 0;
};

template <typename T>
Status SaveTensorWithFixedBuffer(const string& tensor_name,
                                 BundleWriter* writer, char* dump_buffer,
                                 size_t bytes_limit, DumpIterator<T>* dump_iter,
                                 const TensorShape& dump_tensor_shape,
                                 bool use_shape = true) {
  bool dump_happened = false;
  size_t bytes_written = 0;
  int buffer_idx = 0;
  Status st;
  int64 total_bytes_written = 0;
  T* key_dump_buffer = (T*)dump_buffer;
  if (use_shape)
    st = writer->AddTensorHeader(tensor_name, DataTypeToEnum<T>::v(),
                                 dump_tensor_shape);
  if (!st.ok()) return st;

  while (dump_iter->HasNext()) {
    T key = dump_iter->Next();
    if (bytes_written + sizeof(T) > bytes_limit) {
      dump_happened = true;
      TF_CHECK_OK(writer->AppendSegmentData(dump_buffer, bytes_written));
      bytes_written = 0;
      buffer_idx = 0;
    }
    key_dump_buffer[buffer_idx] = key;
    buffer_idx++;
    bytes_written += sizeof(T);
    total_bytes_written += sizeof(T);
  }

  if (!dump_happened) {
    VLOG(1) << tensor_name
            << " only one buffer written, size:" << bytes_written;
    TF_CHECK_OK(writer->AddCompeleteData(dump_buffer, bytes_written));
  } else {
    VLOG(1) << tensor_name
            << " mutiple buffer written, size:" << total_bytes_written
            << ", bytes written:" << bytes_written;
    TF_CHECK_OK(writer->AppendSegmentData(dump_buffer, bytes_written));
    writer->EndSegmentData(total_bytes_written, bytes_written);
  }
  return OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_EV_H_
