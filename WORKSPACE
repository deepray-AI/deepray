workspace(name = "deepray")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")  # buildifier: disable=load-on-top
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")  # buildifier: disable=load-on-top

http_archive(
    name = "rules_python",
    sha256 = "d71d2c67e0bce986e1c5a7731b4693226867c45bfe0b7c5e0067228a536fc580",
    strip_prefix = "rules_python-0.29.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.29.0/rules_python-0.29.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")  # buildifier: disable=load-on-top

py_repositories()

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = "3.10",
)

load("//third_party/xla:workspace.bzl", xla_repo = "repo")

xla_repo()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.10": "//build_deps:requirements_lock_3_10.txt",
        "3.11": "//build_deps:requirements_lock_3_11.txt",
        "3.12": "//build_deps:requirements_lock_3_12.txt",
        "3.13": "//build_deps:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl", "cuda_json_init_repository")

cuda_json_init_repository()

load("@cuda_redist_json//:distributions.bzl", "CUDA_REDISTRIBUTIONS", "CUDNN_REDISTRIBUTIONS")
load("@tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl", "cuda_redist_init_repositories", "cudnn_redist_init_repository")

cuda_redist_init_repositories(cuda_redistributions = CUDA_REDISTRIBUTIONS)

cudnn_redist_init_repository(cudnn_redistributions = CUDNN_REDISTRIBUTIONS)

load("@tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")

load("@tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl", "nccl_redist_init_repository")

nccl_redist_init_repository()

load("@tsl//third_party/nccl/hermetic:nccl_configure.bzl", "nccl_configure")

nccl_configure(name = "local_config_nccl")

load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

http_archive(
    name = "org_tensorflow",
    patch_args = ["-p1"],
    patches = [
        "//third_party/tf:tf_215.patch",
    ],
    sha256 = "f36416d831f06fe866e149c7cd752da410a11178b01ff5620e9f265511ed57cf",
    strip_prefix = "tensorflow-2.15.1",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.15.1.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@//deepray:workspace3.bzl", "dp_workspace3")

dp_workspace3()

load("@//deepray:workspace2.bzl", "dp_workspace2")

dp_workspace2()

# load("@//deepray:workspace1.bzl", "dp_workspace1")

# dp_workspace1()

load("@//deepray:workspace0.bzl", "dp_workspace0")

dp_workspace0()
