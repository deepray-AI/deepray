licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# TODO: Once TF exports symbols in a DLL we can enable parse_time_op for windows
# https://github.com/tensorflow/deepray/issues/782
py_library(
    name = "text",
    srcs = glob(["*.py"]),
    data = select({
        "//deepray:windows": [
            "//deepray/custom_ops/text:_skip_gram_ops.so",
            "//deepray/testing",
            "//deepray/utils",
        ],
        "//conditions:default": [
            "//deepray/custom_ops/text:_parse_time_op.so",
            "//deepray/custom_ops/text:_skip_gram_ops.so",
            "//deepray/testing",
            "//deepray/utils",
        ],
    }),
)

py_test(
    name = "text_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":text",
        "//deepray/layers",
    ],
)
