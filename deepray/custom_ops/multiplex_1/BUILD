# Build multiplex_1 custom ops examples, which is similar to np.where
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "multiplex_1_kernel.so",
    srcs = [
        "multiplex_1_kernel.cc",
        "multiplex_1_op.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
    ],
)

py_library(
    name = "multiplex_1",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [":multiplex_1_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "multiplex_1_test",
    size = "medium",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_1",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
