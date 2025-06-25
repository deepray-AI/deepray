"""Defines the base task abstraction."""

import abc
import functools
from typing import Optional

from absl import logging
import tensorflow as tf


class FrameWork(tf.keras.Model, metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def build_network(self, flags=None, features=None):
    """
    must defined in subclass
    """
    raise NotImplementedError("build_network: not implemented!")

  def build_features(self):
    pass
