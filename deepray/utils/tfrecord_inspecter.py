import tensorflow as tf

file = "/workspaces/dataset/avazu/raw/train_train.tfrecord"

raw_dataset = tf.data.TFRecordDataset(file)

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
