load("//build_deps/pip_tf:defs.bzl", "tf_custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

tf_custom_op_library(
    name = "_skip_gram_ops.so",
    srcs = [
        "cc/kernels/skip_gram_kernels.cc",
        "cc/ops/skip_gram_ops.cc",
    ],
)

tf_custom_op_library(
    name = "_parse_time_op.so",
    srcs = select({
        "//deepray:windows": [],
        "//conditions:default": [
            "cc/kernels/parse_time_kernel.cc",
            "cc/ops/parse_time_op.cc",
        ],
    }),
)
