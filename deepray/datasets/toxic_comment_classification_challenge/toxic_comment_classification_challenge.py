import os
import sys
import zipfile

import pandas as pd
import tensorflow as tf
import texthero as hero
from absl import flags
from sklearn.model_selection import train_test_split
from texthero import preprocessing

from deepray.datasets.datapipeline import DataPipeline

os.environ["CURL_CA_BUNDLE"] = ""

FLAGS([
  sys.argv[0],
  "--num_train_examples=111699",
])


class ToxicCommentClassificationChallenge(DataPipeline):
  def __init__(self, path="/workspaces/dataset/jigsaw-toxic-comment-classification-challenge", **kwargs):
    super().__init__(**kwargs)

    with zipfile.ZipFile(os.path.join(path, "train.csv.zip"), "r") as zip_ref:
      zip_ref.extractall("data")

    df = pd.read_csv(os.path.join(path, "train.csv"))  # .head(10000)

    train, test = train_test_split(df, test_size=0.3, random_state=1)

    clean_text_pipeline = [
      preprocessing.remove_urls,  # remove urls
      preprocessing.remove_punctuation,  # remove punctuation
      preprocessing.remove_digits,  # remove numbers
      preprocessing.remove_diacritics,
      preprocessing.lowercase,  # convert to lowercase
      preprocessing.remove_stopwords,  # remove stopwords
      preprocessing.remove_whitespace,  # remove any extra spaces
      preprocessing.stem,  # stemming of the words
    ]
    # applying the processing pipeline
    train["clean_text"] = hero.clean(train["comment_text"], clean_text_pipeline)
    test["clean_text"] = hero.clean(test["comment_text"], clean_text_pipeline)
    # comparison of text before and after the processing
    print("Original Text:")
    print(train["comment_text"].iloc[0], "\n")
    print("Clean Text:")
    print(train["clean_text"].iloc[0])

    print("------------------------------------")

    print("Original Text:")
    print(train["comment_text"].iloc[1], "\n")
    print("Clean Text:")
    print(train["clean_text"].iloc[1])

    # create targets
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    self.y_train = train[labels]
    self.y_test = test[labels]

    # before feeding the data to the model, we will clean it a bit
    clean_text_bert_pipeline = [
      preprocessing.remove_urls,  # remove urls
      preprocessing.remove_diacritics,
      preprocessing.remove_whitespace,  # remove any extra spaces
    ]
    self.train_bert = hero.clean(train["comment_text"], clean_text_bert_pipeline)
    self.test_bert = hero.clean(test["comment_text"], clean_text_bert_pipeline)

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, *args, **kwargs):
    if is_training:
      ds = tf.data.Dataset.from_tensor_slices((self.train_bert, self.y_train))
    else:
      ds = tf.data.Dataset.from_tensor_slices((self.test_bert, self.y_test))
    ds = ds.repeat(FLAGS.epochs).shuffle(50000).batch(batch_size)
    return ds
