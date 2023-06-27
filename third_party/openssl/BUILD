load("@bazel_skylib//rules:build_test.bzl", "build_test")

exports_files(
    [
        "openssl.BUILD",
    ],
    visibility = ["//visibility:public"],
)

build_test(
    name = "build_test",
    targets = [
        "@openssl//:openssl",
    ],
)
