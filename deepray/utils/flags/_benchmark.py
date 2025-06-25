# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Flags for benchmarking models."""

from absl import flags

from deepray.utils.flags._conventions import help_wrap


def define_log_steps():
  flags.DEFINE_integer(
    name="log_steps", default=100, help="Frequency with which to log timing information with TimeHistory."
  )

  return []


def define_benchmark(bigquery_uploader=False):
  """Register benchmarking flags.

  Args:
    bigquery_uploader: Create flags for uploading results to BigQuery.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []
  flags.DEFINE_enum(
    name="benchmark_logger_type",
    default="BaseBenchmarkLogger",
    enum_values=["BaseBenchmarkLogger", "BenchmarkFileLogger"],
    help=help_wrap(
      "The type of benchmark logger to use. Defaults to using "
      "BaseBenchmarkLogger which logs to STDOUT. Different "
      "loggers will require other flags to be able to work."
    ),
  )
  flags.DEFINE_string(
    name="benchmark_test_id",
    default=None,
    help=help_wrap(
      "The unique test ID of the benchmark run. It could be the "
      "combination of key parameters. It is hardware "
      "independent and could be used compare the performance "
      "between different test runs. This flag is designed for "
      "human consumption, and does not have any impact within "
      "the system."
    ),
  )

  define_log_steps()

  if bigquery_uploader:
    flags.DEFINE_string(
      name="gcp_project", default=None, help=help_wrap("The GCP project name where the benchmark will be uploaded.")
    )

    flags.DEFINE_string(
      name="bigquery_data_set",
      default="test_benchmark",
      help=help_wrap("The Bigquery dataset name where the benchmark will be uploaded."),
    )

    flags.DEFINE_string(
      name="bigquery_run_table",
      default="benchmark_run",
      help=help_wrap("The Bigquery table name where the benchmark run information will be uploaded."),
    )

    flags.DEFINE_string(
      name="bigquery_run_status_table",
      default="benchmark_run_status",
      help=help_wrap("The Bigquery table name where the benchmark run status information will be uploaded."),
    )

    flags.DEFINE_string(
      name="bigquery_metric_table",
      default="benchmark_metric",
      help=help_wrap("The Bigquery table name where the benchmark metric information will be uploaded."),
    )

  return key_flags
