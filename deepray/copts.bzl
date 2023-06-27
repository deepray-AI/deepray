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

"""specific copts.
"""

BASE_COPTS = [
    "-g",
    "-ggdb",
    "-O0",
    "-fno-omit-frame-pointer",
    # "-gno-statement-frontiers",
    # "-gno-variable-location-views",
    "-Wall",
    # "-rdynamic",

    # for lldb
    # https://stackoverflow.com/questions/58578615/cannot-inspect-a-stdstring-variable-in-lldb
    # "-fstandalone-debug",
]

BASE_C_COPTS = BASE_COPTS + [
    # "--std=gnu89",
    "-std=gnu11",
    "-D_GNU_SOURCE",
]

BASE_CPP_COPTS = BASE_COPTS + [
    "-std=c++2a",
]

IGNORED_COPTS = [
    # Note: We cannot require external libraries to conform to specifications
    "-Wno-deprecated-declarations",
    "-Wno-deprecated",
    "-Wno-sign-compare",
    "-Wno-unknown-pragmas",
    "-Wno-unused-but-set-variable",
    "-Wno-unused-function",
    "-Wno-unused-value",
    "-Wno-unused-variable",
    "-Wno-unused-local-typedefs",
    "-Wno-maybe-uninitialized",
]

DEFAULT_C_COPTS = BASE_C_COPTS + IGNORED_COPTS + ["-Wno-implicit-function-declaration"]
DEFAULT_CPP_COPTS = BASE_CPP_COPTS + IGNORED_COPTS

DEFAULT_LINKOPTS = [
    "-latomic",
    "-lpthread",
    "-ldl",
    # "-fuse-ld=lld",
]

TEST_CPP_COPTS = DEFAULT_CPP_COPTS + [
    "-fsanitize=address",
]

TEST_LINKOPTS = DEFAULT_LINKOPTS + [
    "-fsanitize=address",
]
