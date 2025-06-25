load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_config_tf//:build_defs.bzl", "CPLUSPLUS_VERSION", "D_GLIBCXX_USE_CXX11_ABI")
load(
    "@org_tensorflow//tensorflow:py.default.bzl",
    _plain_py_library = "py_library",
)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_binary_additional_srcs", "tf_copts")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_shared_library = native.cc_shared_library

def _cuda_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) CUDA compilation.

    If we're doing CUDA compilation, returns copts for our particular CUDA
    compiler.  If we're not doing CUDA compilation, returns an empty list.

    """
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ] + opts,
        "@local_config_cuda//cuda:using_clang": [
            "-fcuda-flush-denormals-to-zero",
        ] + opts,
    })

def custom_op_library(
        name,
        srcs = [],
        gpu_srcs = [],
        deps = [],
        gpu_deps = [],
        copts = [],
        **kwargs):
    """
    Reference: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/tensorflow_addons.bzl
    """
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
    ]

    if gpu_srcs:
        basename = name.split(".")[0]
        cuda_library(
            name = basename + "_gpu",
            srcs = gpu_srcs,
            copts = copts + tf_copts() + _cuda_copts(),
            deps = deps + gpu_deps,
            **kwargs
        )
        deps = deps + [":" + basename + "_gpu"]

    copts = copts + select({
        "//deepray:windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/D_USE_MATH_DEFINES",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": ["-pthread", CPLUSPLUS_VERSION, D_GLIBCXX_USE_CXX11_ABI],
    })

    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts,
        linkshared = 1,
        features = select({
            "//deepray:windows": ["windows_export_all_symbols"],
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )

def clean_dep(target):
    """Returns string to 'target' in @org_tensorflow repository.

    Use this function when referring to targets in the @org_tensorflow
    repository from macros that may be called from external repositories.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, i.e. @org_tensorflow.
    return str(Label(target))

def filegroup(**kwargs):
    native.filegroup(**kwargs)

def _rpath_user_link_flags(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("@platforms//os:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//deepray:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$ORIGIN", levels_to_root),),
        ],
    })

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("@platforms//os:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//deepray:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

# buildozer: disable=function-docstring-args
def pybind_extension_opensource(
        name,
        srcs,
        module_name = None,  # Unused.
        hdrs = [],
        dynamic_deps = [],
        static_deps = [],
        deps = [],
        additional_exported_symbols = [],
        compatible_with = None,
        copts = [],
        data = [],
        defines = [],
        deprecation = None,
        enable_stub_generation = False,  # Unused.
        additional_stubgen_deps = [],  # Unused.
        features = [],
        link_in_framework = False,
        licenses = None,
        linkopts = [],
        pytype_deps = [],
        pytype_srcs = [],
        restricted_to = None,
        srcs_version = "PY3",
        testonly = None,
        visibility = None,
        win_def_file = None):
    """Builds a generic Python extension module."""
    _ignore = [enable_stub_generation, additional_stubgen_deps, module_name]  # buildifier: disable=unused-variable
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    filegroup_name = "%s_filegroup" % name
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ] + additional_exported_symbols

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    if static_deps:
        cc_library_name = so_file + "_cclib"
        cc_library(
            name = cc_library_name,
            hdrs = hdrs,
            srcs = srcs + hdrs,
            data = data,
            deps = deps,
            compatible_with = compatible_with,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                clean_dep("//deepray:windows"): [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            defines = defines,
            features = features + ["-use_header_modules"],
            restricted_to = restricted_to,
            testonly = testonly,
            visibility = visibility,
        )

        cc_shared_library(
            name = so_file,
            roots = [cc_library_name],
            dynamic_deps = dynamic_deps,
            static_deps = static_deps,
            additional_linker_inputs = [exported_symbols_file, version_script_file],
            compatible_with = compatible_with,
            deprecation = deprecation,
            features = features + ["-use_header_modules"],
            licenses = licenses,
            restricted_to = restricted_to,
            shared_lib_name = so_file,
            testonly = testonly,
            user_link_flags = linkopts + _rpath_user_link_flags(name) + select({
                clean_dep("@platforms//os:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//deepray:windows"): [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            visibility = visibility,
        )

        # cc_shared_library can generate more than one file.
        # Solution to avoid the error "variable '$<' : more than one input file."
        filegroup(
            name = filegroup_name,
            srcs = [so_file],
            output_group = "main_shared_library_output",
            testonly = testonly,
        )
    else:
        if link_in_framework:
            srcs += tf_binary_additional_srcs()

        cc_binary(
            name = so_file,
            srcs = srcs + hdrs,
            data = data,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                clean_dep("//deepray:windows"): [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            linkopts = linkopts + _rpath_linkopts(name) + select({
                clean_dep("@platforms//os:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//deepray:windows"): [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            deps = deps + [
                exported_symbols_file,
                version_script_file,
            ],
            defines = defines,
            features = features + ["-use_header_modules"],
            linkshared = 1,
            testonly = testonly,
            licenses = licenses,
            visibility = visibility,
            deprecation = deprecation,
            restricted_to = restricted_to,
            compatible_with = compatible_with,
        )

        # For Windows, emulate the above filegroup with the shared object.
        native.alias(
            name = filegroup_name,
            actual = so_file,
        )

    # For Windows only.
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [filegroup_name],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
        testonly = testonly,
    )

    _plain_py_library(
        name = name,
        data = select({
            clean_dep("//deepray:windows"): [pyd_file],
            "//conditions:default": [so_file],
        }) + pytype_srcs,
        deps = pytype_deps,
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )

# Export open source version of pybind_extension under base name as well.
pybind_extension = pybind_extension_opensource
