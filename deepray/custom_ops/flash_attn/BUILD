load("//deepray:deepray.bzl", "custom_op_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")

licenses(["notice"])  # Apache 2.0

cuda_library(
    name = "tf_flash_attn",
    srcs = if_cuda([
        "cc/ops/flash_attention_ops.cc",
        "cc/kernels/flash_attention_fwd_kernel.cc",
        "cc/kernels/flash_attention_bwd_kernel.cc",
    ]),
    hdrs = if_cuda([
        "cc/kernels/flash_attention.h",
    ]),
    copts = [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    deps = if_cuda([
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@flash_attn//:flash_attn",
    ]),
)

custom_op_library(
    name = "_flash_attn_ops.so",
    cuda_deps = [
        "@flash_attn//:flash_attn",
    ],
    cuda_srcs = [
        "cc/kernels/flash_attention.h",
        "cc/kernels/flash_attention_bwd_kernel.cc",
        "cc/kernels/flash_attention_fwd_kernel.cc",
        "cc/ops/flash_attention_ops.cc",
    ],
)

py_library(
    name = "flash_attn",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_flash_attn_ops.so",
        "//deepray:options.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//deepray/utils",
    ],
)
