"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@bazel_skylib//lib:versions.bzl", "versions")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/nccl:nccl_configure.bzl", "nccl_configure")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//deepray/tools/toolchains:cpus/aarch64/aarch64_compiler_configure.bzl", "aarch64_compiler_configure")
load("//deepray/tools/toolchains:cpus/arm/arm_compiler_configure.bzl", "arm_compiler_configure")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("//deepray/tools/def_file_filter:def_file_filter_configure.bzl", "def_file_filter_configure")
load("//third_party/llvm:setup.bzl", "llvm_setup")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("//third_party/gemmlowp:workspace.bzl", gemmlowp = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/libprotobuf_mutator:workspace.bzl", libprotobuf_mutator = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/pybind11_abseil:workspace.bzl", pybind11_abseil = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/pasta:workspace.bzl", pasta = "repo")
load("//third_party/psimd:workspace.bzl", psimd = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party/sobol_data:workspace.bzl", sobol_data = "repo")
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("//third_party/tensorrt:workspace.bzl", tensorrt = "repo")
load("//third_party/triton:workspace.bzl", triton = "repo")

# Import external repository rules.
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@tf_runtime//:dependencies.bzl", "tfrt_dependencies")
load("//deepray/tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")
load("//deepray/tools/toolchains/remote:configure.bzl", "remote_execution_configure")
load("//deepray/tools/toolchains/clang6:repo.bzl", "clang6_configure")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    bazel_skylib_workspace()
    benchmark()
    dlpack()
    farmhash()
    gemmlowp()
    highwayhash()
    hwloc()
    kissfft()
    libprotobuf_mutator()
    nasm()
    opencl_headers()
    pasta()
    psimd()
    pybind11_abseil()
    ruy()
    sobol_data()
    stablehlo()
    tensorrt()
    triton()

    # copybara: tsl vendor

# Toolchains & platforms required by Tensorflow to build.
def _tf_toolchains():
    native.register_execution_platforms("@local_execution_config_platform//:platform")
    native.register_toolchains("@local_execution_config_python//:py_toolchain")

    # Loads all external repos to configure RBE builds.
    initialize_rbe_configs()

    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    cuda_configure(name = "local_config_cuda")
    tensorrt_configure(name = "local_config_tensorrt")
    nccl_configure(name = "local_config_nccl")
    git_configure(name = "local_config_git")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    rocm_configure(name = "local_config_rocm")
    remote_execution_configure(name = "local_config_remote_execution")

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = "//deepray/tools/toolchains/cpus/arm:template.BUILD",
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # Load aarch64 toolchain
    aarch64_compiler_configure()

# Define all external repositories required by TensorFlow
def _tf_repositories():
    """All external dependencies for TF builds."""

    # To update any of the dependencies below:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "5de2d0d3dc9b4d23e93f51c1e441d2cc245e590e698bc186ed98f677bf39aef5",
        strip_prefix = "XNNPACK-7adae8e6ded8fff33d92212ca1028d2419cd34d4",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/7adae8e6ded8fff33d92212ca1028d2419cd34d4.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)

    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "b96413b10dd8edaa4f6c0a60c6cf5ef55eebeef78164d5d69294c8173457f0ec",
        strip_prefix = "pthreadpool-b8374f80e42010941bda6c85b0e3f1a1bd77a1e0",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip"),
    )

    tf_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-3dc310302210c1891ffcfb12ae67b11a3ad3a150",
        sha256 = "ba668f9f8ea5b4890309b7db1ed2e152aaaf98af6f9a8a63dbe1b75c04e52cb9",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/3dc310302210c1891ffcfb12ae67b11a3ad3a150.zip"),
    )

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "bfcf778030831f325cfc13ae5995388cc834fbff2995a297ba580d9ec65ca3b6",
        strip_prefix = "cudnn-frontend-0.8",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v0.8.zip"),
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        sha256 = "a50993aa6265b799b040fe745e0010502f9f7103cc53a9525d59646aef006633",
        strip_prefix = "oneDNN-2.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:onednn_acl_threadcap.patch",
            "//third_party/mkl_dnn:onednn_acl_fixed_format_kernels.patch",
            "//third_party/mkl_dnn:onednn_acl_depthwise_convolution.patch",
            "//third_party/mkl_dnn:onednn_acl_threadpool_scheduler.patch",
        ],
        sha256 = "a50993aa6265b799b040fe745e0010502f9f7103cc53a9525d59646aef006633",
        strip_prefix = "oneDNN-2.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/v2.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "compute_library",
        sha256 = "e20a060d3c4f803889d96c2f0b865004ba3ef4e228299a44339ea1c1ba827c85",
        strip_prefix = "ComputeLibrary-22.11",
        build_file = "//third_party/compute_library:BUILD",
        patch_file = [
            "//third_party/compute_library:compute_library.patch",
            "//third_party/compute_library:acl_fixed_format_kernels_striding.patch",
            "//third_party/compute_library:acl_openmp_fix.patch",
        ],
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/v22.11.tar.gz"),
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = "//:arm_compiler.BUILD",
        sha256 = "b9e7d50ffd9996ed18900d041d362c99473b382c0ae049b2fce3290632d2656f",
        strip_prefix = "rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/",
        urls = tf_mirror_urls("https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz"),
    )

    tf_http_archive(
        # This is the latest `aarch64-none-linux-gnu` compiler provided by ARM
        # See https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
        # The archive contains GCC version 9.2.1
        name = "aarch64_compiler",
        build_file = "//:arm_compiler.BUILD",
        sha256 = "8dfe681531f0bd04fb9c53cf3c0a3368c616aa85d48938eebe2b516376e06a66",
        strip_prefix = "gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "aarch64_linux_toolchain",
        build_file = "//deepray/tools/toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD",
        sha256 = "50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = "//deepray/tools/toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD",
        sha256 = "3f76650b1d048036473b16b647b8fd005ffccd1a2869c10994967e0e49f26ac2",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf.tar.xz"),
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "b90430b2a9240df4459108b3e291be80ae92c68a47bc06ef2dc419c5724de061",
        strip_prefix = "re2-a276a8c738735a0fe45a6ee590fe2df69bcf4502",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/a276a8c738735a0fe45a6ee590fe2df69bcf4502.tar.gz"),
    )

    tf_http_archive(
        name = "png",
        build_file = "//third_party:png.BUILD",
        patch_file = ["//third_party:png_fix_rpi.patch"],
        sha256 = "a00e9d2f2f664186e4202db9299397f851aea71b36a35e74910b8820e380d441",
        strip_prefix = "libpng-1.6.39",
        system_build_file = "//third_party/systemlibs:png.BUILD",
        urls = tf_mirror_urls("https://github.com/glennrp/libpng/archive/v1.6.39.tar.gz"),
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party:sqlite.BUILD",
        sha256 = "49112cc7328392aa4e3e5dae0b2f6736d0153430143d21f69327788ff4efe734",
        strip_prefix = "sqlite-amalgamation-3400100",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = tf_mirror_urls("https://www.sqlite.org/2022/sqlite-amalgamation-3400100.zip"),
    )

    tf_http_archive(
        name = "gif",
        build_file = "//third_party:gif.BUILD",
        patch_file = ["//third_party:gif_fix_strtok_r.patch"],
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = "//third_party/systemlibs:gif.BUILD",
        urls = tf_mirror_urls("https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz"),
    )

    tf_http_archive(
        name = "six_archive",
        build_file = "//third_party:six.BUILD",
        sha256 = "1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926",
        strip_prefix = "six-1.16.0",
        system_build_file = "//third_party/systemlibs:six.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/s/six/six-1.16.0.tar.gz"),
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = "//third_party:astor.BUILD",
        sha256 = "95c30d87a6c2cf89aa628b87398466840f0ad8652f88eb173125a6df8533fb8d",
        strip_prefix = "astor-0.7.1",
        system_build_file = "//third_party/systemlibs:astor.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/a/astor/astor-0.7.1.tar.gz"),
    )

    tf_http_archive(
        name = "astunparse_archive",
        build_file = "//third_party:astunparse.BUILD",
        sha256 = "5ad93a8456f0d084c3456d059fd9a92cce667963232cbf763eac3bc5b7940872",
        strip_prefix = "astunparse-1.6.3/lib",
        system_build_file = "//third_party/systemlibs:astunparse.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz"),
    )

    tf_http_archive(
        name = "functools32_archive",
        build_file = "//third_party:functools32.BUILD",
        sha256 = "f6253dfbe0538ad2e387bd8fdfd9293c925d63553f5813c4e587745416501e6d",
        strip_prefix = "functools32-3.2.3-2",
        system_build_file = "//third_party/systemlibs:functools32.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz"),
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = "//third_party:gast.BUILD",
        sha256 = "40feb7b8b8434785585ab224d1568b857edb18297e5a3047f1ba012bc83b42c1",
        strip_prefix = "gast-0.4.0",
        system_build_file = "//third_party/systemlibs:gast.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/83/4a/07c7e59cef23fb147454663c3271c21da68ba2ab141427c20548ae5a8a4d/gast-0.4.0.tar.gz"),
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = "//third_party:termcolor.BUILD",
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = "//third_party/systemlibs:termcolor.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz"),
    )

    tf_http_archive(
        name = "typing_extensions_archive",
        build_file = "//third_party:typing_extensions.BUILD",
        sha256 = "f1c24655a0da0d1b67f07e17a5e6b2a105894e6824b92096378bb3668ef02376",
        strip_prefix = "typing_extensions-4.2.0/src",
        system_build_file = "//third_party/systemlibs:typing_extensions.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/t/typing_extensions/typing_extensions-4.2.0.tar.gz"),
    )

    tf_http_archive(
        name = "opt_einsum_archive",
        build_file = "//third_party:opt_einsum.BUILD",
        sha256 = "d3d464b4da7ef09e444c30e4003a27def37f85ff10ff2671e5f7d7813adac35b",
        strip_prefix = "opt_einsum-2.3.2",
        system_build_file = "//third_party/systemlibs:opt_einsum.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz"),
    )

    http_archive(
        name = "com_google_absl",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.0.tar.gz",
        ],
        strip_prefix = "abseil-cpp-20220623.0",
        sha256 = "4208129b49006089ba1d6710845a45e31c59b0ab6bff9e5788a87f55c5abd602",
    )

    tf_http_archive(
        name = "dill_archive",
        build_file = "//third_party:dill.BUILD",
        system_build_file = "//third_party/systemlibs:dill.BUILD",
        urls = tf_mirror_urls("https://github.com/uqfoundation/dill/releases/download/dill-0.3.6/dill-0.3.6.zip"),
        sha256 = "2159ca9e7568ff47dc7be2e35a6edf18014351da95ad1b59c0930a14dcf37be7",
        strip_prefix = "dill-0.3.6",
    )

    tf_http_archive(
        name = "tblib_archive",
        build_file = "//third_party:tblib.BUILD",
        system_build_file = "//third_party/systemlibs:tblib.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/d3/41/901ef2e81d7b1e834b9870d416cb09479e175a2be1c4aa1a9dcd0a555293/tblib-1.7.0.tar.gz"),
        sha256 = "059bd77306ea7b419d4f76016aef6d7027cc8a0785579b5aad198803435f882c",
        strip_prefix = "tblib-1.7.0",
    )

    PROTOBUF_VERSION = "3.21.9"
    git_repository(
        name = "com_google_protobuf",
        remote = "https://github.com/protocolbuffers/protobuf",
        tag = "v" + PROTOBUF_VERSION,
    )

    tf_http_archive(
        name = "nsync",
        patch_file = ["//third_party:nsync.patch"],
        sha256 = "2be9dbfcce417c7abcc2aa6fee351cd4d292518d692577e74a2c6c05b049e442",
        strip_prefix = "nsync-1.25.0",
        system_build_file = "//third_party/systemlibs:nsync.BUILD",
        urls = tf_mirror_urls("https://github.com/google/nsync/archive/1.25.0.tar.gz"),
    )

    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"],
        strip_prefix = "googletest-release-1.12.1",
    )

    tf_http_archive(
        name = "com_google_fuzztest",
        sha256 = "c75f224b34c3c62ee901381fb743f6326f7b91caae0ceb8fe62f3fd36f187627",
        strip_prefix = "fuzztest-58b4e7065924f1a284952b84ea827ce35a87e4dc",
        urls = tf_mirror_urls("https://github.com/google/fuzztest/archive/58b4e7065924f1a284952b84ea827ce35a87e4dc.zip"),
    )

    git_repository(
        name = "com_github_gflags_gflags",
        remote = "https://github.com/gflags/gflags.git",
        tag = "v2.2.2",
    )

    tf_http_archive(
        name = "curl",
        build_file = "//third_party:curl.BUILD",
        sha256 = "dfb8582a05a893e305783047d791ffef5e167d295cf8d12b9eb9cfa0991ca5a9",
        strip_prefix = "curl-7.88.0",
        system_build_file = "//third_party/systemlibs:curl.BUILD",
        urls = tf_mirror_urls("https://curl.haxx.se/download/curl-7.88.0.tar.gz"),
    )

    tf_http_archive(
        name = "linenoise",
        build_file = "//third_party:linenoise.BUILD",
        sha256 = "b35a74dbc9cd2fef9e4d56222761d61daf7e551510e6cd1a86f0789b548d074e",
        strip_prefix = "linenoise-4ce393a66b10903a0ef52edf9775ed526a17395f",
        urls = tf_mirror_urls("https://github.com/antirez/linenoise/archive/4ce393a66b10903a0ef52edf9775ed526a17395f.tar.gz"),
    )

    llvm_setup(name = "llvm-project")

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD",
        patch_file = ["//third_party/llvm_openmp:openmp_switch_default_patch.patch"],
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = tf_mirror_urls("https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"),
    )

    tf_http_archive(
        name = "jsoncpp_git",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
        system_build_file = "//third_party/systemlibs:jsoncpp.BUILD",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"),
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "9dc53f851107eaf87b391136d13b815df97ec8f76dadb487b58b2fc45e624d2c",
        strip_prefix = "boringssl-c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc",
        system_build_file = "//third_party/systemlibs:boringssl.BUILD",
        urls = tf_mirror_urls("https://github.com/google/boringssl/archive/c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc.tar.gz"),
    )

    # LINT.IfChange
    tf_http_archive(
        name = "fft2d",
        build_file = "//third_party/fft2d:fft2d.BUILD",
        sha256 = "5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb",
        strip_prefix = "OouraFFT-1.0",
        urls = tf_mirror_urls("https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/fft2d.cmake)

    tf_http_archive(
        name = "nccl_archive",
        build_file = "//third_party:nccl/archive.BUILD",
        patch_file = ["//third_party/nccl:archive.patch"],
        sha256 = "0e3d7b6295beed81dc15002e88abf7a3b45b5c686b13b779ceac056f5612087f",
        strip_prefix = "nccl-2.16.5-1",
        urls = tf_mirror_urls("https://github.com/nvidia/nccl/archive/v2.16.5-1.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_pprof",
        build_file = "//third_party:pprof.BUILD",
        sha256 = "b844b75c25cfe7ea34b832b369ab91234009b2dfe2ae1fcea53860c57253fe2e",
        strip_prefix = "pprof-83db2b799d1f74c40857232cb5eb4c60379fe6c2",
        urls = tf_mirror_urls("https://github.com/google/pprof/archive/83db2b799d1f74c40857232cb5eb4c60379fe6c2.tar.gz"),
    )

    # The CUDA 11 toolkit ships with CUB.  We should be able to delete this rule
    # once TF drops support for CUDA 10.
    tf_http_archive(
        name = "cub_archive",
        build_file = "//third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = tf_mirror_urls("https://github.com/NVlabs/cub/archive/1.9.9.zip"),
    )

    tf_http_archive(
        name = "nvtx_archive",
        build_file = "//third_party:nvtx.BUILD",
        sha256 = "bb8d1536aad708ec807bc675e12e5838c2f84481dec4005cd7a9bbd49e326ba1",
        strip_prefix = "NVTX-3.0.1/c/include",
        urls = tf_mirror_urls("https://github.com/NVIDIA/NVTX/archive/v3.0.1.tar.gz"),
    )

    tf_http_archive(
        name = "cython",
        build_file = "//third_party:cython.BUILD",
        sha256 = "08dbdb6aa003f03e65879de8f899f87c8c718cd874a31ae9c29f8726da2f5ab0",
        strip_prefix = "cython-3.0.0a11",
        system_build_file = "//third_party/systemlibs:cython.BUILD",
        urls = tf_mirror_urls("https://github.com/cython/cython/archive/3.0.0a11.tar.gz"),
    )

    # LINT.IfChange
    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = "//third_party:arm_neon_2_x86_sse.BUILD",
        sha256 = "019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d",
        strip_prefix = "ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff",
        urls = tf_mirror_urls("https://github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/neon2sse.cmake)

    http_archive(
        name = "com_github_google_double_conversion",
        urls = ["https://github.com/google/double-conversion/archive/v3.2.0.tar.gz"],
        build_file = Label("//third_party:double_conversion.BUILD"),
        sha256 = "3dbcdf186ad092a8b71228a5962009b5c96abde9a315257a3452eb988414ea3b",
        strip_prefix = "double-conversion-3.2.0",
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz"),
    )

    # https://github.com/google/xctestrunner/releases
    tf_http_archive(
        name = "xctestrunner",
        strip_prefix = "xctestrunner-0.2.15",
        sha256 = "b789cf18037c8c28d17365f14925f83b93b1f7dabcabb80333ae4331cf0bcb2f",
        urls = tf_mirror_urls("https://github.com/google/xctestrunner/archive/refs/tags/0.2.15.tar.gz"),
    )

    tf_http_archive(
        name = "nlohmann_json_lib",
        build_file = "//third_party:nlohmann_json.BUILD",
        sha256 = "5daca6ca216495edf89d167f808d1d03c4a4d929cef7da5e10f135ae1540c7e4",
        strip_prefix = "json-3.10.5",
        urls = tf_mirror_urls("https://github.com/nlohmann/json/archive/v3.10.5.tar.gz"),
    )

    tf_http_archive(
        name = "pybind11",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz"),
        sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
        strip_prefix = "pybind11-2.10.0",
        build_file = "//third_party:pybind11.BUILD",
        system_build_file = "//third_party/systemlibs:pybind11.BUILD",
    )

    tf_http_archive(
        name = "pybind11_protobuf",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/80f3440cd8fee124e077e2e47a8a17b78b451363.zip"),
        sha256 = "c7ab64b1ccf9a678694a89035a8c865a693e4e872803778f91f0965c2f281d78",
        strip_prefix = "pybind11_protobuf-80f3440cd8fee124e077e2e47a8a17b78b451363",
        patch_file = ["//third_party/pybind11_protobuf:remove_license.patch"],
    )

    tf_http_archive(
        name = "wrapt",
        build_file = "//third_party:wrapt.BUILD",
        sha256 = "866211ed43c2639a2452cd017bd38589e83687b1d843817c96b99d2d9d32e8d7",
        strip_prefix = "wrapt-1.14.1/src/wrapt",
        system_build_file = "//third_party/systemlibs:wrapt.BUILD",
        urls = tf_mirror_urls("https://github.com/GrahamDumpleton/wrapt/archive/1.14.1.tar.gz"),
    )

    tf_http_archive(
        name = "coremltools",
        sha256 = "89bb0bd2c16e19932670838dd5a8b239cd5c0a42338c72239d2446168c467a08",
        strip_prefix = "coremltools-5.2",
        build_file = "//third_party:coremltools.BUILD",
        urls = tf_mirror_urls("https://github.com/apple/coremltools/archive/5.2.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_glog_glog",
        sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
        strip_prefix = "glog-0.4.0",
        urls = tf_mirror_urls("https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_ortools",
        sha256 = "b87922b75bbcce9b2ab5da0221751a3c8c0bff54b2a1eafa951dbf70722a640e",
        strip_prefix = "or-tools-7.3",
        patch_file = ["//third_party/ortools:ortools.patch"],
        urls = tf_mirror_urls("https://github.com/google/or-tools/archive/v7.3.tar.gz"),
        repo_mapping = {"@com_google_protobuf_cc": "@com_google_protobuf"},
    )

    tf_http_archive(
        name = "glpk",
        sha256 = "9a5dab356268b4f177c33e00ddf8164496dc2434e83bd1114147024df983a3bb",
        build_file = "//third_party/ortools:glpk.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
            "http://ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
        ],
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
        # sha256 = "57e13c62f27b710e1de54fd30faed612aefa22aa41fa2c0c3bacd204dd18a8f3",
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

    FBTHRIFT_VERSION = "2022.01.24.00"
    http_archive(
        name = "com_github_facebook_fbthrift",
        build_file = Label("//third_party/fbthrift:fbthrift.BUILD"),
        # sha256 = "0fc6cc1673209f4557e081597b2311f6c9f153840c4e55ac61a669e10207e2ee",
        strip_prefix = "fbthrift-" + FBTHRIFT_VERSION,
        url = "https://github.com/facebook/fbthrift/archive/v{}.tar.gz".format(FBTHRIFT_VERSION),
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
        name = "openssl",
        build_file = Label("//third_party/openssl:openssl.BUILD"),
        sha256 = "892a0875b9872acd04a9fde79b1f943075d5ea162415de3047c327df33fbaee5",
        strip_prefix = "openssl-1.1.1k",
        urls = [
            "https://mirror.bazel.build/www.openssl.org/source/openssl-1.1.1k.tar.gz",
            "https://www.openssl.org/source/openssl-1.1.1k.tar.gz",
            "https://github.com/openssl/openssl/archive/OpenSSL_1_1_1k.tar.gz",
        ],
    )

    http_archive(
        name = "aliyun_oss_c_sdk",
        build_file = "//third_party:oss_c_sdk.BUILD",
        sha256 = "6450d3970578c794b23e9e1645440c6f42f63be3f82383097660db5cf2fba685",
        strip_prefix = "aliyun-oss-c-sdk-3.7.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
            "https://github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
        ],
    )

    http_archive(
        name = "avro",
        build_file = "//third_party:avro.BUILD",
        sha256 = "8fd1f850ce37e60835e6d8335c0027a959aaa316773da8a9660f7d33a66ac142",
        strip_prefix = "avro-release-1.10.1/lang/c++",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/avro/archive/release-1.10.1.tar.gz",
            "https://github.com/apache/avro/archive/release-1.10.1.tar.gz",
        ],
    )

    http_archive(
        name = "aws-checksums",
        build_file = "//third_party:aws-checksums.BUILD",
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
    )

    http_archive(
        name = "aws-c-common",
        build_file = "//third_party:aws-c-common.BUILD",
        sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
        strip_prefix = "aws-c-common-0.4.29",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
            "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        ],
    )

    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party:aws-c-event-stream.BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
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
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aws/aws-sdk-cpp/archive/1.8.186.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.8.186.tar.gz",
        ],
    )

    http_archive(
        name = "jemalloc",
        # sha256 = "ed51b0b37098af4ca6ed31c22324635263f8ad6471889e0592a9c0dba9136aea",
        strip_prefix = "jemalloc-5.3.0",
        build_file = Label("//third_party:jemalloc.BUILD"),
        urls = ["https://github.com/jemalloc/jemalloc/archive/5.3.0.tar.gz"],
    )

def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    versions.check("1.0.0")

    # Initialize toolchains and platforms.
    _tf_toolchains()

    # Import third party repositories according to go/tfbr-thirdparty.
    _initialize_third_party()

    # Import all other repositories. This should happen before initializing
    # any external repositories, because those come with their own
    # dependencies. Those recursive dependencies will only be imported if they
    # don't already exist (at least if the external repository macros were
    # written according to common practice to query native.existing_rule()).
    _tf_repositories()

    tfrt_dependencies()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace2 = workspace
