"""Flags which will be nearly universal across models."""

from absl import flags
from deepray.utils.flags import core as flags_core
from deepray.utils.flags._conventions import help_wrap


def define_data_download_flags(
  dataset=False,
  data_dir=False,
  download_if_missing=False,
):
  """Add flags specifying data download and usage arguments."""
  key_flags = []
  if dataset:
    flags.DEFINE_string(
      "dataset", default=None, help=flags_core.help_wrap("The name of the dataset, e.g. ImageNet, etc.")
    )
    key_flags.append("dataset")
  if data_dir:
    flags.DEFINE_string(
      name="data_dir",
      default="/tmp/movielens-data/",
      help=flags_core.help_wrap("Directory to download and extract data."),
    )
    key_flags.append("data_dir")
  if download_if_missing:
    flags.DEFINE_boolean(
      name="download_if_missing",
      default=True,
      help=flags_core.help_wrap("Download data to data_dir if it is not already present."),
    )
    key_flags.append("download_if_missing")
  return key_flags
