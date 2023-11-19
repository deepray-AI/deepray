load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])


exports_files(["LICENSE"])

cc_library(
    name = "cuco_hash_table",
    hdrs = glob(["include/**"]),
    includes = [
        "include",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    visibility = ["//visibility:public"],
)
