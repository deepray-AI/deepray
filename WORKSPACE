load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")

tf_configure(
    name = "local_config_tf",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//deepray:workspace3.bzl", "dp_workspace3")

dp_workspace3()

load("@//deepray:workspace2.bzl", "dp_workspace2")

dp_workspace2()

load("@//deepray:workspace1.bzl", "dp_workspace1")

dp_workspace1()

load("@//deepray:workspace0.bzl", "dp_workspace0")

dp_workspace0()
