"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# buildifier: disable=unnamed-macro
def workspace(with_rules_cc = True):
    """Loads a set of TensorFlow dependencies. To be used in a WORKSPACE file.

    Args:
      with_rules_cc: whether to load and patch rules_cc repository.
    """
    native.register_toolchains("@local_config_python//:py_toolchain")

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace1 = workspace
