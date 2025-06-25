# Build multiplex_3 custom ops example, which is similar to np.where.
# This example shows how a Python wrapper can choose either to use "dispach
# for custom object types" to choose an old C++ Op (that supports only dense
# tensors) for backwards compatibility or a new C++ for new functionality
# (that supprots sparse tensors).

load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "multiplex_3_kernel.so",
    srcs = [
        "multiplex_3_kernel.cc",
        "multiplex_3_op.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
    ],
)

py_library(
    name = "multiplex_3_op",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [":multiplex_3_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//deepray/custom_ops/multiplex_2:multiplex_2_op",
    ],
)

py_test(
    name = "multiplex_3_test",
    size = "small",
    srcs = ["multiplex_3_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_3_op",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
