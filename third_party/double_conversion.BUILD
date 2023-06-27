# vim: ft=bzl
load("@//deepray:copts.bzl", "DEFAULT_CPP_COPTS")

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "double-conversion",
    srcs = glob(["double-conversion/*.cc"]),
    hdrs = glob(["double-conversion/*.h"]),
    copts = DEFAULT_CPP_COPTS,
    includes = ["."],
    visibility = ["//visibility:public"],
)
