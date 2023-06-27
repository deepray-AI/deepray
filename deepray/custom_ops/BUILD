licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "custom_ops",
    srcs = glob(["**/*.py"]),
    # srcs = glob(["*.py"]),
    deps = [
        "//deepray/custom_ops/correlation_cost",
        "//deepray/custom_ops/embedding_bag",
        "//deepray/custom_ops/image",
        "//deepray/custom_ops/parquet_dataset",
        "//deepray/custom_ops/text",
    ],
)

py_test(
    name = "custom_ops_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":custom_ops",
    ],
)
