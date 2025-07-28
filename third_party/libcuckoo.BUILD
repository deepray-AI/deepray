load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "libcuckoo",
    build_args = [
        "--verbose",
        "-j `nproc`",
    ],
    lib_source = ":all_srcs",
    out_headers_only = True,
)
