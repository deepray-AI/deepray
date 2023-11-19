package(default_visibility = ["//visibility:public"])

py_library(
    name = "custom_ops",
    srcs = glob(["**/*.py"]),
    deps = [
        # "//deepray/custom_ops/correlation_cost",
        "//deepray/custom_ops/embedding_bag",
        # "//deepray/custom_ops/flash_attn",
        "//deepray/custom_ops/multiplex_1:multiplex_1_op",
        "//deepray/custom_ops/multiplex_2:multiplex_2_op",
        "//deepray/custom_ops/multiplex_3:multiplex_3_op",
        "//deepray/custom_ops/multiplex_4:multiplex_4_op",
        "//deepray/custom_ops/parquet_dataset",
        "//deepray/custom_ops/simple_hash_table",
        "//deepray/custom_ops/sleep:sleep_op",
        "//deepray/custom_ops/training_ops",
        "//deepray/custom_ops/zero_out:zero_out_ops",
    ],
)
