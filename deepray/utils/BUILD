licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    data = [
        "//deepray:conftest.py",
        "//deepray:options.py",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":utils",
    ],
)
