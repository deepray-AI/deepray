load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make", "configure_make_variant")

# Read https://wiki.openssl.org/index.php/Compilation_and_Installation

filegroup(
    name = "all_srcs",
    srcs = glob(
        include = ["**"],
        exclude = ["*.bazel"],
    ),
)

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

configure_make(
    name = "openssl",
    args = [
        "-j `nproc`",
    ],
    configure_command = "config",
    configure_in_place = True,
    configure_options = [
        "no-comp",
        "no-idea",
        "no-weak-ssl-ciphers",
        "no-shared",
    ],
    env = select({
        "@platforms//os:macos": {
            "AR": "",
        },
        "//conditions:default": {
        },
    }),
    lib_name = "openssl",
    lib_source = ":all_srcs",
    out_lib_dir = "lib64",
    # Note that for Linux builds, libssl must come before libcrypto on the linker command-line.
    # As such, libssl must be listed before libcrypto
    #out_shared_libs = ["libssl.so.1.1", "libcrypto.so.1.1",],
    out_static_libs = [
        "libssl.a",
        "libcrypto.a",
    ],
    targets = [
        "build_libs",
        "install_dev",
        "build_programs",
        "install_sw",
    ],
    visibility = ["//visibility:public"],
)
