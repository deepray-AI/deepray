#!/usr/bin/env python
# @Time    : 2021/12/7 8:44 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import importlib
import os
import sys

from absl import logging, flags

from deepray.design_patterns import SingletonType


class InputMeta(metaclass=SingletonType):
  def __init__(self, conf_version=None):
    self.conf_version = conf_version if conf_version else FLAGS.input_meta_data_path
    self.conf = self.import_conf()

  def import_conf(self):
    project_path = os.path.abspath(os.curdir)
    conf_path = os.path.join(project_path, "examples/Recommendation/CGC/conf")
    logging.info(conf_path)

    def file_name(file_dir, target):
      paths = []
      for root, dirs, files in os.walk(file_dir):
        basename = os.path.basename(root)
        if basename == target:
          paths.append(root)
      if len(paths) == 0:
        logging.info(f"Cannot find conf: {target}")
        sys.exit()
      elif len(paths) > 1:
        logging.info("Found more than one conf:")
        for path in paths:
          logging.info(" ", os.path.relpath(path, project_path))
        sys.exit()
      else:
        return paths[0]

    local_conf_path = file_name(conf_path, self.conf_version)
    conf_module = os.path.relpath(local_conf_path, project_path).replace("/", ".")
    conf = importlib.import_module(f"{conf_module}.params", "*")
    # import examples.Recommendation.CGC.conf.default_geek_predict.conf_geek_predict_mix_4_target_cgc_f2_1v8_weight_tfra_base_new.params as conf
    return conf
