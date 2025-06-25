#!/usr/bin/env python
# @Time    : 2021/8/10 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
from absl import flags
from deepray.datasets.mnist import Mnist

data_pipe = Mnist()
# create data pipline of train & test dataset
train_dataset = data_pipe(batch_size=flags.FLAGS.batch_size, is_training=True)
test_dataset = data_pipe(batch_size=flags.FLAGS.batch_size, is_training=False)

num_examples = 0
for x in train_dataset:
  num_examples += flags.FLAGS.batch_size

print(num_examples)
