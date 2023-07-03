load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["LICENSE"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "xsimd",
    build_args = [
        "-j `nproc`",
    ],
    lib_source = ":all_srcs",
    out_headers_only = True,
)
