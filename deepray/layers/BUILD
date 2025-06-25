package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "layers",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    deps = [
        "//deepray/activations",
        "//deepray/layers/rnn",
        "//deepray/testing",
        # "//deepray/text",
        "//deepray/utils",
        "@pypi_pandas//:pkg",
    ],
)

# py_test(
#     name = "layers_test",
#     size = "large",
#     srcs = glob(["tests/*"]),
#     main = "tests/run_all_test.py",
#     deps = [
#         ":layers",
#     ],
# )
