import os
import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds

from models import Transformer
import deepray as dp
from deepray.core.trainer import Trainer
from deepray.optimizers import lamb

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
# os.environ['XLA_FLAGS'] = '--xla_gpu_enable_cudnn_frontend --xla_gpu_enable_cudnn_fmha'

BUFFER_SIZE = 20000
BATCH_SIZE = 64


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match) / tf.reduce_sum(mask)


MAX_TOKENS = 128

# model_name = 'ted_hrlr_translate_pt_en_converter'
# tf.keras.utils.get_file(
#   f'{model_name}.zip',
#   f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
#   cache_dir='.', cache_subdir='', extract=True
# )


def main():
  tokenizers = tf.saved_model.load("/workspaces/datasets/ted_hrlr_translate_pt_en_converter")

  def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels

  def make_batches(ds):
    return (
        ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(prepare_batch,
                                                      tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    )

  transformer = Transformer(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
      target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
      dropout_rate=dropout_rate
  )

  learning_rate = CustomSchedule(d_model)
  # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  optimizer = lamb.LAMB(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  trainer = Trainer(
      model=transformer,
      loss=masked_loss,
      optimizer=optimizer,
      metrics=[masked_accuracy],
      # jit_compile=True
  )

  examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

  train_examples, val_examples = examples['train'], examples['validation']
  # Create training and validation set batches.
  train_batches = make_batches(train_examples)
  val_batches = make_batches(val_examples)

  trainer.fit(
      train_input=train_batches,
      # steps_per_epoch=811,
      # eval_input=val_batches,
      # eval_steps=valid_steps,
      # callbacks=[ModelCheckpoint()],
  )


if __name__ == "__main__":
  dp.runner(main)
