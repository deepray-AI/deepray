/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef DEEPRAY_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_
#define DEEPRAY_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 16
#include "unsupported/Eigen/CXX11/Tensor"
#else
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif

namespace tensorflow {
class OpKernelContext;

namespace deepray {

namespace functor {

template <typename Device, typename T>
struct GatherTree {
  void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<T, 3>::ConstTensor step_ids,
                  typename TTypes<T, 3>::ConstTensor parent_ids,
                  TTypes<int32>::ConstVec max_sequence_lengths,
                  const T end_token, typename TTypes<T, 3>::Tensor beams);
};

}  // namespace functor
}  // end namespace deepray
}  // namespace tensorflow

#endif  // DEEPRAY_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_
