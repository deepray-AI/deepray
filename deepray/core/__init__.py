import tensorflow as tf


class ExportModel(tf.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  @tf.function
  def __call__(self, imgs):
    return self.model(imgs, training=False, post_mode='global')
