load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["LICENSE"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "cuCollections",
    build_args = [
        "-j `nproc`",
    ],
    cache_entries = {
        "BUILD_TESTS": "OFF",
        "BUILD_BENCHMARKS": "OFF",
        "BUILD_EXAMPLES": "OFF",
    },
    lib_source = ":all_srcs",
    out_headers_only = True,
)
