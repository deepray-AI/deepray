licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//deepray:deepray.bzl", "custom_op_library")

custom_op_library(
    name = "_zero_out_ops.so",
    srcs = [
        "cc/kernels/zero_out_kernels.cc",
        "cc/ops/zero_out_ops.cc",
    ]
)

py_library(
    name = "zero_out_ops_py",
    srcs = ([
        "python/ops/zero_out_ops.py",
    ]),
    data = [
        ":_zero_out_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "zero_out_ops_py_test",
    srcs = [
        "python/ops/zero_out_ops_test.py"
    ],
    main = "python/ops/zero_out_ops_test.py",
    deps = [
        ":zero_out_ops_py",
        "//deepray/utils",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "zero_out_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":zero_out_ops_py"
    ],
    srcs_version = "PY2AND3",
)
