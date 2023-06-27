sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_deps/build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "requirements.txt",
        "setup.py",
        "//deepray",
    ],
)

load("@bazel_skylib//rules:build_test.bzl", "build_test")

build_test(
    name = "build_test",
    targets = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@eigen3",
    ],
)
