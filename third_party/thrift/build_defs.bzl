# Copyright 2021 curoky(cccuroky@gmail.com).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rules for building C++ thrift with Bazel.
"""

thriftc_path = "@com_github_apache_thrift//:thriftc"

DEFAULT_INCLUDE_PATHS = [
    "./",
    "$(GENDIR)",
    "$(BINDIR)",
]

def thrift_library(
        name,
        srcs,
        services = [],
        language = "cpp",
        out_prefix = ".",
        include_prefix = ".",
        include_paths = [],
        thriftc_args = [],
        visibility = ["//visibility:public"]):
    output_headers = []
    output_srcs = []
    for s in srcs:
        fname = s.replace(".thrift", "").split("/")[-1]
        output_headers.append("%s_constants.h" % fname)
        output_srcs.append("%s_constants.cpp" % fname)

        output_headers.append("%s_types.h" % fname)
        output_srcs.append("%s_types.cpp" % fname)

    for sname in services:
        output_headers.append("%s.h" % sname)
        output_srcs.append("%s.cpp" % sname)

    _output_files = output_headers + output_srcs
    output_files = []
    for f in _output_files:
        if out_prefix and out_prefix != ".":
            output_files.append(out_prefix + "/gen-cpp/" + f)
        else:
            output_files.append("gen-cpp/" + f)

    include_paths_cmd = ["-I %s" % (s) for s in include_paths]

    # '$(@D)' when given a single source target will give the appropriate
    # directory. Appending 'out_prefix' is only necessary when given a build
    # target with multiple sources.
    # output_directory = (
    #     ("-o $(@D)/%s" % (out_prefix)) if len(srcs) > 1 else ("-o $(@D)")
    # )
    output_directory = "-o $(@D)/%s" % (out_prefix)

    # -gen mstch_cpp2:include_prefix=thrift/lib/thrift -o $(@D)/thrift/lib/thrift $<
    genrule_cmd = " ".join([
        "mkdir -p $(@D)/%s;" % (out_prefix),
        "SRCS=($(SRCS));",
        "for f in $${SRCS[@]:0:%s}; do" % len(srcs),
        "$(location %s)" % (thriftc_path),
        "-gen %s:include_prefix=%s" % (language, include_prefix),
        " ".join(thriftc_args),
        " ".join(include_paths_cmd),
        output_directory,
        "$$f;",
        "done",
    ])

    native.genrule(
        name = name,
        srcs = srcs,
        outs = output_files,
        output_to_bindir = False,
        tools = [thriftc_path],
        cmd = genrule_cmd,
        message = "Generating thrift files for %s:" % (name),
        visibility = visibility,
    )
