licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "feature_map",
    srcs = glob([
        "**/feature_map.csv",
    ]),
)

py_library(
    name = "datapipeline",
    srcs = ["datapipeline.py"],
    deps = [
        "//deepray",
        "//deepray/utils",
    ],
)

py_library(
    name = "datasets",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    data = [
        ":feature_map",
    ],
    deps = [
        "//deepray/testing",
        "//deepray/utils",
    ],
)

# py_test(
#     name = "datasets_test",
#     size = "small",
#     srcs = glob(["tests/*"]),
#     main = "tests/run_all_test.py",
#     deps = [
#         ":datasets",
#     ],
# )
