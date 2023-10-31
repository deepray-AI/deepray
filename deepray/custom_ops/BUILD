package(default_visibility = ["//visibility:public"])

py_library(
    name = "custom_ops",
    srcs = glob(["**/*.py"]),
    deps = [
        "//deepray/custom_ops/correlation_cost",
        "//deepray/custom_ops/parquet_dataset",
        "//deepray/custom_ops/zero_out:zero_out_ops",
    ],
)
