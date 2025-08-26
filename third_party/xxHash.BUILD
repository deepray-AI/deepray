load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "xxHash",
    build_args = [
        "--verbose",
        "-j `nproc`",
    ],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
    },
    lib_source = ":all_srcs",
    out_static_libs = [
        "libxxhash.a",
    ],
    working_directory = "cmake_unofficial",
)
