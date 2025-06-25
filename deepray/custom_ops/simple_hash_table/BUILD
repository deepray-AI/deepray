# Build simple_hash_table custom ops example, which is similar to,
# but simpler than, tf.lookup.experimental.MutableHashTable

load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
    ],
)

py_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
)

py_library(
    name = "simple_hash_table",
    # srcs = [
    #     "__init__.py",
    #     "simple_hash_table.py",
    #     "simple_hash_table_op.py",
    # ],
    srcs = glob(
        [
            "*.py",
        ],
    ),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":simple_hash_table_op",
    ],
)

py_test(
    name = "simple_hash_table_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["simple_hash_table_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":simple_hash_table",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
