licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//deepray:deepray.bzl", "custom_op_library")
load("//deepray:tensorflow.bzl", "tf_cuda_library")

custom_op_library(
    name = "_dot_based_interact_ops.so",
    srcs = [
        "cc/kernels/dot_based_interact_grad_kernels.cc",
        "cc/kernels/dot_based_interact_kernels.cc",
        "cc/ops/dot_based_interact_ops.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "cc/kernels/ampere/dot_based_interact_ampere.h",
        "cc/kernels/ampere/dot_based_interact_ampere.cu",
        "cc/kernels/cuda_kernels/dot_based_interact_fp16.cu",
        "cc/kernels/cuda_kernels/dot_based_interact_fp32.cu",
        "cc/kernels/cuda_kernels/dot_based_interact_shared_utils.cuh",
        "cc/kernels/cuda_kernels/dot_based_interact_tf32.cu",
        "cc/kernels/launchers/dot_based_interact_fp16_launcher.cu",
        "cc/kernels/launchers/dot_based_interact_fp32_launcher.cu",
        "cc/kernels/launchers/dot_based_interact_tf32_launcher.cu",
        "cc/kernels/volta/dot_based_interact_volta.cu",
        "cc/kernels/volta/dot_based_interact_volta.h",
    ],
    deps = [
        # "@com_github_apache_arrow//:arrow",
        # "@eigen3",
        # "@local_config_tf//:libtensorflow_framework",
        # "@local_config_tf//:tf_header_lib",
    ],
)

tf_cuda_library(
    name = "cuda_kernels",
    srcs = [
        "cc/kernels/cuda_kernels/dot_based_interact_fp16.cu",
        "cc/kernels/cuda_kernels/dot_based_interact_fp32.cu",
        "cc/kernels/cuda_kernels/dot_based_interact_tf32.cu",
    ],
    hdrs = ["cc/kernels/cuda_kernels/dot_based_interact_shared_utils.cuh"],
)
