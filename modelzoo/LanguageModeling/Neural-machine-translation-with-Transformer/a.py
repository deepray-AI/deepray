import logging
import time

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

examples, metadata = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True)

train_examples, val_examples = examples["train"], examples["validation"]

for pt_examples, en_examples in train_examples.batch(3).take(1):
  print("> Examples in Portuguese:")
  for pt in pt_examples.numpy():
    print(pt.decode("utf-8"))
  print()

  print("> Examples in English:")
  for en in en_examples.numpy():
    print(en.decode("utf-8"))

model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
  f"{model_name}.zip",
  f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
  cache_dir=".",
  cache_subdir="",
  extract=True,
)

tokenizers = tf.saved_model.load(model_name)

[item for item in dir(tokenizers.en) if not item.startswith("_")]

print("> This is a batch of strings:")
for en in en_examples.numpy():
  print(en.decode("utf-8"))

encoded = tokenizers.en.tokenize(en_examples)

print("> This is a padded-batch of token IDs:")
for row in encoded.to_list():
  print(row)

round_trip = tokenizers.en.detokenize(encoded)

print("> This is human-readable text:")
for line in round_trip.numpy():
  print(line.decode("utf-8"))

print("> This is the text split into tokens:")
tokens = tokenizers.en.lookup(encoded)
print(tokens)

embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

pt_emb = embed_pt(pt)
en_emb = embed_en(en)
sample_ca = CrossAttention(num_heads=2, key_dim=512)

print(pt_emb.shape)
print(en_emb.shape)
print(sample_ca(en_emb, pt_emb).shape)
