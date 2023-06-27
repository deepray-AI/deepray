# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//deepray:__subpackages__"],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    deps = [
        "//deepray",
    ],
)
