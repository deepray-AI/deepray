load("//deepray:deepray.bzl", "custom_op_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

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
    deps = [
        "//deepray/utils",
    ],
)
