from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepray as dp
from absl import app, flags
from tensorflow import keras

from deepray.core.trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.cifar import CIFAR100
from .model import BaseModel

learning_rate = 1e-3
batch_size = 128
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  data_pipe = CIFAR100()
  with distribution_utils.get_strategy_scope(_strategy):
    model = BaseModel(
        input_shape=(32, 32, 3),
        patch_size=(2, 2),  # 2-by-2 sized patches
        dropout_rate=0.03,  # Dropout rate
        num_heads=8,  # Attention heads
        embed_dim=64,  # Embedding dimension
        num_mlp=256,  # MLP layer size
        qkv_bias=True,  # Convert embedded patches to query, key, and values with a learnable additive value
        window_size=2,  # Size of attention window
        shift_size=1,  # Size of shifting window
        image_dimension=32,  # Initial image size
    )(num_classes=100)

  trainer = Trainer(
      model=model,
      loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
      optimizer=dp.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
      metrics=[
          keras.metrics.CategoricalAccuracy(name="accuracy"),
          keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
      ],
  )

  train_input_fn = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  trainer.fit(train_input=train_input_fn,)

  # trainer.export_tfra()
  """
  Let's display the final results of the training on CIFAR-100.
  """

  # loss, accuracy, top_5_accuracy = trainer.evaluate(x_test, y_test)
  # print(f"Test loss: {round(loss, 2)}")
  # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
  # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
  """
  The Swin Transformer model we just trained has just 152K parameters, and it gets
  us to ~75% test top-5 accuracy within just 40 epochs without any signs of overfitting
  as well as seen in above graph. This means we can train this network for longer
  (perhaps with a bit more regularization) and obtain even better performance.
  This performance can further be improved by additional techniques like cosine
  decay learning rate schedule, other data augmentation techniques. While experimenting,
  I tried training the model for 150 epochs with a slightly higher dropout and greater
  embedding dimensions which pushes the performance to ~72% test accuracy on CIFAR-100
  as you can see in the screenshot.

  ![Results of training for longer](https://i.imgur.com/9vnQesZ.png)

  The authors present a top-1 accuracy of 87.3% on ImageNet. The authors also present
  a number of experiments to study how input sizes, optimizers etc. affect the final
  performance of this model. The authors further present using this model for object detection,
  semantic segmentation and instance segmentation as well and report competitive results
  for these. You are strongly advised to also check out the
  [original paper](https://arxiv.org/abs/2103.14030).

  This example takes inspiration from the official
  [PyTorch](https://github.com/microsoft/Swin-Transformer) and
  [TensorFlow](https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow) implementations.
  """


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
