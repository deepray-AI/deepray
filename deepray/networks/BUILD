licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "networks",
    srcs = glob(["*.py",
                 "**/*.py"]),
    deps = [
        "//deepray/testing",
        "//deepray/utils",
    ],
)

py_test(
    name = "networks_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":networks",
    ],
)
