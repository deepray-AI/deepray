"""Deepray workspace initialization. Consult the WORKSPACE on how to use it."""

# Import external repository rules.
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Define all external repositories required by TensorFlow
def _tf_repositories():
    """All external dependencies for Deepray builds."""

    # To update any of the dependencies below:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    http_archive(
        name = "com_github_google_double_conversion",
        urls = ["https://github.com/google/double-conversion/archive/v3.2.0.tar.gz"],
        build_file = Label("//third_party:double_conversion.BUILD"),
        sha256 = "3dbcdf186ad092a8b71228a5962009b5c96abde9a315257a3452eb988414ea3b",
        strip_prefix = "double-conversion-3.2.0",
    )

    git_repository(
        name = "rules_python",
        remote = "https://github.com/bazelbuild/rules_python.git",
        tag = "0.16.2",
    )

    http_archive(
        name = "eigen3",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/{tag}/eigen-{tag}.tar.gz".format(tag = "3.4.0"),
        ],
        strip_prefix = "eigen-{}".format("3.4.0"),
        build_file = Label("//third_party:eigen3.BUILD"),
    )

    OPENBLAS_VERSION = "0.3.23"
    http_archive(
        name = "openblas",
        urls = [
            "https://github.com/xianyi/OpenBLAS/releases/download/v{tag}/OpenBLAS-{tag}.tar.gz".format(tag = OPENBLAS_VERSION),
        ],
        type = "tar.gz",
        strip_prefix = "OpenBLAS-{}".format(OPENBLAS_VERSION),
        # sha256 = "947f51bfe50c2a0749304fbe373e00e7637600b0a47b78a51382aeb30ca08562",
        build_file = Label("//third_party:openblas.BUILD"),
    )

    FLAT_VERSION = "1.12.0"
    http_archive(
        name = "com_github_google_flatbuffers",
        urls = [
            "https://github.com/google/flatbuffers/archive/v{}.tar.gz".format(FLAT_VERSION),
        ],
        strip_prefix = "flatbuffers-" + FLAT_VERSION,
        # sha256 = "d84cb25686514348e615163b458ae0767001b24b42325f426fd56406fd384238",
    )

    ARROW_VERSION = "7.0.0"
    http_archive(
        name = "com_github_apache_arrow",
        sha256 = "57e13c62f27b710e1de54fd30faed612aefa22aa41fa2c0c3bacd204dd18a8f3",
        build_file = Label("//third_party/arrow:arrow.BUILD"),
        strip_prefix = "arrow-apache-arrow-" + ARROW_VERSION,
        urls = [
            "https://github.com/apache/arrow/archive/apache-arrow-{}.tar.gz".format(ARROW_VERSION),
        ],
    )

    # git_repository(
    #     name = "com_github_google_brotli",
    #     remote = "https://github.com/google/brotli",
    #     tag = "v1.0.9",
    # )
    http_archive(
        name = "com_github_google_brotli",  # MIT license
        build_file = Label("//third_party:brotli.BUILD"),
        sha256 = "4c61bfb0faca87219ea587326c467b95acb25555b53d1a421ffa3c8a9296ee2c",
        strip_prefix = "brotli-1.0.7",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/brotli/archive/v1.0.7.tar.gz",
            "https://github.com/google/brotli/archive/v1.0.7.tar.gz",
        ],
    )

    XSMID_VERSION = "11.0.0"
    http_archive(
        name = "com_github_xtensorstack_xsimd",
        urls = [
            "https://github.com/xtensor-stack/xsimd/archive/refs/tags/{}.tar.gz".format(XSMID_VERSION),
        ],
        strip_prefix = "xsimd-" + XSMID_VERSION,
        build_file = Label("//third_party:xsimd.BUILD"),
    )

    http_archive(
        name = "com_github_tencent_rapidjson",
        urls = [
            "https://github.com/Tencent/rapidjson/archive/{}.tar.gz".format("00dbcf2c6e03c47d6c399338b6de060c71356464"),
        ],
        strip_prefix = "rapidjson-" + "00dbcf2c6e03c47d6c399338b6de060c71356464",
        build_file = Label("//third_party:rapidjson.BUILD"),
        sha256 = "b4339b8118d57f70de7a17ed8f07997080f98940ca538f43e1ca4b95a835221d",
    )

    # new_git_repository(
    #     name = "com_github_apache_thrift",
    #     remote = "https://github.com/apache/thrift",
    #     branch = "0.16.0",
    #     build_file = Label("//third_party/thrift:thrift.BUILD"),
    # )

    http_archive(
        name = "com_github_apache_thrift",  # Apache License 2.0
        build_file = Label("//third_party/thrift:thrift.BUILD"),
        sha256 = "5da60088e60984f4f0801deeea628d193c33cec621e78c8a43a5d8c4055f7ad9",
        strip_prefix = "thrift-0.13.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/thrift/archive/v0.13.0.tar.gz",
            "https://github.com/apache/thrift/archive/v0.13.0.tar.gz",
        ],
    )

    # Boost
    # Famous C++ library that has given rise to many new additions to the C++ Standard Library
    # Makes @boost available for use: For example, add `@boost//:algorithm` to your deps.
    # For more, see https://github.com/nelhage/rules_boost and https://www.boost.org
    http_archive(
        name = "com_github_nelhage_rules_boost",
        # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
        # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
        url = "https://github.com/nelhage/rules_boost/archive/96e9b631f104b43a53c21c87b01ac538ad6f3b48.tar.gz",
        strip_prefix = "rules_boost-96e9b631f104b43a53c21c87b01ac538ad6f3b48",
        # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'com_github_nelhage_rules_boost' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
    )

    http_archive(
        name = "aliyun_oss_c_sdk",
        build_file = "//third_party:oss_c_sdk.BUILD",
        sha256 = "6450d3970578c794b23e9e1645440c6f42f63be3f82383097660db5cf2fba685",
        strip_prefix = "aliyun-oss-c-sdk-3.7.0",
        urls = [
            "https://github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
        ],
    )

    http_archive(
        name = "aws-checksums",
        build_file = "//third_party:aws-checksums.BUILD",
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        urls = [
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
    )

    http_archive(
        name = "aws-c-common",
        build_file = "//third_party:aws-c-common.BUILD",
        sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
        strip_prefix = "aws-c-common-0.4.29",
        urls = [
            "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        ],
    )

    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party:aws-c-event-stream.BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
    )

    http_archive(
        name = "aws-sdk-cpp",
        build_file = "//third_party:aws-sdk-cpp.BUILD",
        patch_cmds = [
            """sed -i.bak 's/UUID::RandomUUID/Aws::Utils::UUID::RandomUUID/g' aws-cpp-sdk-core/source/client/AWSClient.cpp""",
            """sed -i.bak 's/__attribute__((visibility("default")))//g' aws-cpp-sdk-core/include/aws/core/external/tinyxml2/tinyxml2.h """,
        ],
        sha256 = "749322a8be4594472512df8a21d9338d7181c643a00e08a0ff12f07e831e3346",
        strip_prefix = "aws-sdk-cpp-1.8.186",
        urls = [
            "https://github.com/aws/aws-sdk-cpp/archive/1.8.186.tar.gz",
        ],
    )

    PB_COMMIT = "b162c7c88a253e3f6b673df0c621aca27596ce6b"

    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{}".format(PB_COMMIT),
        urls = ["https://github.com/pybind/pybind11_bazel/archive/{}.zip".format(PB_COMMIT)],
    )

    # We still require the pybind library.
    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-2.10.4",
        urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.4.tar.gz"],
    )

    git_repository(
        name = "com_github_google_boringssl",
        commit = "f7f897f45dcc46501b89e6fb3f5338428977ece2",
        remote = "https://boringssl.googlesource.com/boringssl",
    )

def workspace():
    # Import all other repositories. This should happen before initializing
    # any external repositories, because those come with their own
    # dependencies. Those recursive dependencies will only be imported if they
    # don't already exist (at least if the external repository macros were
    # written according to common practice to query native.existing_rule()).
    _tf_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace2 = workspace
