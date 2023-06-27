load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_dot_based_interact_ops.so",
    srcs = [
        "cc/kernels/ampere/dot_based_interact_ampere.h",
        "cc/kernels/cuda_kernels/dot_based_interact_shared_utils.h",
        "cc/kernels/dot_based_interact_grad_kernels.cc",
        "cc/kernels/dot_based_interact_kernels.cc",
        "cc/kernels/volta/dot_based_interact_volta.h",
        "cc/ops/dot_based_interact_ops.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "cc/kernels/ampere/dot_based_interact_ampere.cu.cc",
        "cc/kernels/cuda_kernels/dot_based_interact_fp16.cu.cc",
        "cc/kernels/cuda_kernels/dot_based_interact_fp32.cu.cc",
        "cc/kernels/cuda_kernels/dot_based_interact_tf32.cu.cc",
        "cc/kernels/launchers/dot_based_interact_fp16_launcher.cu.cc",
        "cc/kernels/launchers/dot_based_interact_fp32_launcher.cu.cc",
        "cc/kernels/launchers/dot_based_interact_tf32_launcher.cu.cc",
        "cc/kernels/volta/dot_based_interact_volta.cu.cc",
    ],
    defines = [
        "GOOGLE_CUDA",
    ],
)

py_library(
    name = "dot_based_interact",
    srcs = [
        "python/dot_based_interact_ops.py",
    ],
    data = [
        ":_dot_based_interact_ops.so",
        "//deepray:options.py",
    ],
    deps = [
        "//deepray/utils",
    ],
)

py_test(
    name = "dot_based_interact_test",
    srcs = [
        "python/dot_based_interact_ops_test.py",
    ],
    main = "python/dot_based_interact_ops_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":dot_based_interact",
    ],
)
