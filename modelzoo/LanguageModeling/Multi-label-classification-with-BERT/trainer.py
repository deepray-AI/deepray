import tensorflow as tf
import tensorflow_hub as hub
from absl import app
from absl import flags

from deepray.core.base_trainer import Trainer
from deepray.core.common import distribution_utils
from deepray.datasets.toxic_comment_classification_challenge import ToxicCommentClassificationChallenge

# if FLAGS.use_dynamic_embedding:

FLAGS = flags.FLAGS

# tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1'
tfhub_handle_encoder = 'https://hub.tensorflow.google.cn/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1'

# tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_preprocess = 'https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_preprocess/3'


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(500, activation='relu')(net)
  net = tf.keras.layers.Dense(6, activation="sigmoid", name='classifier')(net)
  return tf.keras.Model(text_input, net)


def main(_):
  _strategy = distribution_utils.get_distribution_strategy()
  with distribution_utils.get_strategy_scope(_strategy):
    model = build_classifier_model()

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint('best_bert_model', save_best_only=True),
      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
  ]
  trainer = Trainer(
      model_or_fn=model,
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[tf.metrics.BinaryAccuracy(), tf.metrics.AUC(multi_label=True)],
  )
  data_pipe = ToxicCommentClassificationChallenge()
  train_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=True)
  valid_dataset = data_pipe(FLAGS.train_data, FLAGS.batch_size, is_training=False)

  trainer.fit(train_input=train_dataset, eval_input=valid_dataset, callbacks=callbacks)


#   trainer.run_evaluation(train_dataset)

if __name__ == "__main__":
  app.run(main)
