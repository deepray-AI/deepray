load("@local_config_cuda//cuda:build_defs.bzl", "cuda_header_library", "if_cuda")

licenses(["notice"])  # # BSD 3-Clause

package(default_visibility = ["//visibility:public"])

cuda_header_library(
    name = "cutlass",
    hdrs = if_cuda(glob([
        "include/cutlass/**",
    ])),
    includes = if_cuda([
        "include",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
