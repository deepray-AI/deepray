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
        name = "com_google_absl",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
        ],
        strip_prefix = "abseil-cpp-20211102.0",
        sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
    )

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
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    )

    OPENBLAS_VERSION = "0.3.25"
    http_archive(
        name = "com_github_OpenMathLib_OpenBLAS",
        urls = [
            "https://github.com/OpenMathLib/OpenBLAS/releases/download/v{tag}/OpenBLAS-{tag}.tar.gz".format(tag = OPENBLAS_VERSION),
        ],
        type = "tar.gz",
        strip_prefix = "OpenBLAS-{}".format(OPENBLAS_VERSION),
        build_file = Label("//third_party:openblas.BUILD"),
        # sha256 = "5d9491d07168a5d00116cdc068a40022c3455bf9293c7cb86a65b1054d7e5114",
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
        sha256 = "50c31c319c8b36c8946eb954c7cca2e2ece86bf8a66a7ebf321b24cd273e7c47",
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
        sha256 = "5ea00abc70cdf396a23fb53201db19ebce2837d28887a08544429d27783309ed",
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

    http_archive(
        name = "cutlass",
        urls = ["https://github.com/NVIDIA/cutlass/archive/319a389f42b776fae5701afcb943fc03be5b5c25.zip"],
        build_file = "//third_party:cutlass.BUILD",
        strip_prefix = "cutlass-319a389f42b776fae5701afcb943fc03be5b5c25",
    )

    http_archive(
        name = "flash_attn",
        urls = ["https://github.com/Dao-AILab/flash-attention/archive/9818f85fee29ac6b60c9214bce841f8109a18b1b.zip"],  # v1.0.4
        build_file = "//third_party/flash_attn:flash_attn.BUILD",
        sha256 = "15f29a1095600ba2a3af688fa96a0a48635edb90fffec56c6eb7c48a4a322d2b",
        strip_prefix = "flash-attention-9818f85fee29ac6b60c9214bce841f8109a18b1b",
        patches = [
            "//third_party/flash_attn:flash_attn.patch",
        ],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "tbb",
        build_file = "//third_party:tbb.BUILD",
        sha256 = "e75fafb171fcd392fdedac14f1a6d6c6211230c6a38169a0ec279ea0d80b8a22",
        strip_prefix = "oneTBB-2019_U1",
        urls = [
            "https://github.com/01org/tbb/archive/2019_U1.zip",
        ],
    )

    http_archive(
        name = "libcuckoo",
        build_file = "//third_party:libcuckoo.BUILD",
        patch_args = ["-p1"],
        patches = [
            "//third_party:cuckoohash_map.patch",
        ],
        sha256 = "7238436b7346a0edf4ce57c12f43f71af5347b8b15f9bf2f0e24bfdca6225fc5",
        strip_prefix = "libcuckoo-0.3",
        urls = [
            "https://github.com/efficient/libcuckoo/archive/v0.3.zip",
        ],
    )

    http_archive(
        name = "sparsehash",
        build_file = "//third_party:sparsehash.BUILD",
        sha256 = "d4a43cad1e27646ff0ef3a8ce3e18540dbcb1fdec6cc1d1cb9b5095a9ca2a755",
        strip_prefix = "sparsehash-c11-2.11.1",
        urls = [
            "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz",
        ],
    )

    http_archive(
        name = "murmurhash",
        build_file = "//third_party:murmurhash.BUILD",
        sha256 = "19a7ccc176ca4185db94047de6847d8a0332e8f4c14e8e88b9048f74bdafe879",
        strip_prefix = "smhasher-master",
        urls = [
            "https://github.com/aappleby/smhasher/archive/master.zip",
        ],
    )

    http_archive(
        name = "farmhash",
        sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",
        build_file = "//third_party:farmhash.BUILD",
        strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
        urls = [
            "https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
            "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        ],
    )

    http_archive(
        name = "cuCollections",  # Apache License 2.0
        # patches = ["//third_party/cucollection:cucollection.patch"],
        build_file = "//third_party/cucollection:cuco.BUILD",
        sha256 = "c5c77a1f96b439b67280e86483ce8d5994aa4d14b7627b1d3bd7880be6be23fa",
        strip_prefix = "cuCollections-193de1aa74f5721717f991ca757dc610c852bb17",
        urls = [
            "https://github.com/NVIDIA/cuCollections/archive/193de1aa74f5721717f991ca757dc610c852bb17.zip",
        ],
    )

    http_archive(
        name = "sparsehash_c11",
        build_file = "//third_party:sparsehash_c11.BUILD",
        sha256 = "d4a43cad1e27646ff0ef3a8ce3e18540dbcb1fdec6cc1d1cb9b5095a9ca2a755",
        strip_prefix = "sparsehash-c11-2.11.1",
        urls = [
            "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz",
        ],
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
