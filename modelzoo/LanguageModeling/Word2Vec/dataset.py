import tensorflow as tf
import tqdm


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      sequence, vocabulary_size=vocab_size, sampling_table=sampling_table, window_size=window_size, negative_samples=0
    )

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=context_class,
        num_true=1,
        num_sampled=num_ns,
        unique=True,
        range_max=vocab_size,
        seed=seed,
        name="negative_sampling",
      )

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0] * num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
