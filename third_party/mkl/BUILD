licenses(["notice"])  # 3-Clause BSD

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

alias(
    name = "build_with_mkl",
    actual = "//deepray/tsl/mkl:build_with_mkl",
)

alias(
    name = "build_with_mkl_lnx_x64",
    actual = "//deepray/tsl/mkl:build_with_mkl_lnx_x64",
)

alias(
    name = "build_with_mkl_lnx_openmp",
    actual = "//deepray/tsl/mkl:build_with_mkl_lnx_openmp",
)

alias(
    name = "build_with_mkl_windows_openmp",
    actual = "//deepray/tsl/mkl:build_with_mkl_windows_openmp",
)

alias(
    name = "build_with_mkl_aarch64",
    actual = "//deepray/tsl/mkl:build_with_mkl_aarch64",
)

alias(
    name = "enable_mkl",
    actual = "//deepray/tsl/mkl:enable_mkl",
)

alias(
    name = "intel_binary_blob",
    actual = "//deepray/tsl/mkl:intel_binary_blob",
)

alias(
    name = "LICENSE",
    actual = "//deepray/tsl/mkl:LICENSE",
)

alias(
    name = "mkl_libs_linux",
    actual = "//deepray/tsl/mkl:mkl_libs_linux",
)

alias(
    name = "mkl_libs_darwin",
    actual = "//deepray/tsl/mkl:mkl_libs_darwin",
)

alias(
    name = "mkl_libs_windows",
    actual = "//deepray/tsl/mkl:mkl_libs_windows",
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
