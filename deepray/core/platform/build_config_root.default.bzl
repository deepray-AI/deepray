"""TODO(jakeharmon): Write module docstring."""

# unused in TSL
def tf_additional_plugin_deps():
    return select({
        str(Label("//deepray/tsl:with_xla_support")): [
            str(Label("//deepray/compiler/jit")),
        ],
        "//conditions:default": [],
    })

def if_dynamic_kernels(extra_deps, otherwise = []):
    return select({
        str(Label("//deepray:dynamic_loaded_kernels")): extra_deps,
        "//conditions:default": otherwise,
    })
