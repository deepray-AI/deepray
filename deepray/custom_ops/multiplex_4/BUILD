# Build multiplex_4 custom ops examples, which is similar to np.where

load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "multiplex_4_kernel.so",
    srcs = [
        "multiplex_4_kernel.cc",
        "multiplex_4_op.cc",
    ],
)

py_library(
    name = "multiplex_4_op",
    srcs = ["multiplex_4_op.py"],
    data = ["multiplex_4_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_using_multiplex",
    srcs = ["model_using_multiplex.py"],
    srcs_version = "PY3",
)

py_test(
    name = "multiplex_4_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    srcs = ["multiplex_4_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":model_using_multiplex",
        ":multiplex_4_op",
    ],
)

py_library(
    name = "multiplex_2_save",
    srcs = ["multiplex_2_save.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "//deepray/custom_ops/multiplex_2:multiplex_2_op",
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "multiplex_4_load_use",
    srcs = ["multiplex_4_load_use.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "multiplex_4_op",
        "@absl_py//absl:app",
    ],
)
