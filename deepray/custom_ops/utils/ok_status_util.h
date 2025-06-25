/* Copyright 2024 The Deepray Authors. All Rights Reserved.

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

#ifndef DEEPRAY_UTILS_H_
#define DEEPRAY_UTILS_H_

// #define PRINT_MACRO_HELPER(x) #x
// #define PRINT_MACRO(x) #x "=" PRINT_MACRO_HELPER(x)

namespace tensorflow {
namespace deepray {

/* After TensorFlow version 2.10.0, "Status::OK()" upgraded to "OkStatus()".
This code is for compatibility.*/
#if TF_VERSION_INTEGER >= 2150
#define TFOkStatus absl::OkStatus()
// #pragma message(PRINT_MACRO(TF_VERSION_INTEGER))
#elif TF_VERSION_INTEGER >= 2100
#define TFOkStatus OkStatus()
// #pragma message(PRINT_MACRO(TF_VERSION_INTEGER))
#else
// #pragma message(PRINT_MACRO(TF_VERSION_INTEGER))
// #define TFOkStatus Status::OK()
#define TFOkStatus absl::OkStatus()
#endif
}  // namespace deepray
}  // namespace tensorflow

#endif  // DEEPRAY_UTILS_H_
