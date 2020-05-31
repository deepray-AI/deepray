#  -*- coding: utf-8 -*-
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================

"""
Embedding layers

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
import tensorflow as tf


class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, layer_name, input_dim, output_dim, dropout=0, initial_range=None, name_scope='embedding',
                 combiner='mean',
                 **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.layer_name = layer_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.initial_range = initial_range
        self.scope_name = name_scope
        self.combiner = combiner

    def build(self, input_shape):
        """
        initialize embedding_weights, where
        id 0 is reserved for UNK, and its embedding fix to all zeros
        """
        with tf.compat.v1.variable_scope(self.scope_name):
            unknown_id = tf.Variable(
                tf.zeros_initializer()([1, self.output_dim]),
                name='-'.join([self.layer_name, 'unknown']),
                trainable=False)
            normal_ids = tf.compat.v1.get_variable(
                '-'.join([self.layer_name, 'normal']),
                [self.input_dim - 1, self.output_dim],
                initializer=tf.random_uniform_initializer(
                    minval=-self.initial_range, maxval=self.initial_range)
                if self.initial_range else None)
        # with tf.name_scope(self.scope_name):
        #     unknown_id = self.add_weight(name='-'.join([self.layer_name, 'unknown']),
        #                                  shape=(1, self.output_dim),
        #                                  initializer=tf.zeros_initializer(),
        #                                  trainable=False)
        #     normal_ids = self.add_weight(name='-'.join([self.layer_name, 'normal']),
        #                                  shape=(self.input_dim - 1, self.output_dim),
        #                                  initializer=tf.random_uniform_initializer(
        #                                      minval=-self.initial_range,
        #                                      maxval=self.initial_range) if self.initial_range else None)
        self.embeddings = tf.concat([unknown_id, normal_ids], axis=0)

    def call(self, inputs, combiner=None, **kwargs):
        """
        input ids shape: Batch_size x 1
        output shape: Batch_size x Embedding_size
        dropout: may tie with the unknown_id ratio
        """
        with tf.name_scope(self.scope_name):
            if isinstance(inputs, tf.SparseTensor):
                out = tf.nn.embedding_lookup_sparse(self.embeddings, inputs, None,
                                                    combiner=combiner)
            else:
                safe_ids = self.safe_ids_for_emb(inputs, self.input_dim)
                out = tf.nn.embedding_lookup(self.embeddings, safe_ids)
                out = tf.reshape(out, [-1, self.output_dim])
            if self.dropout > 0:
                '''
                drop or keep the embedding for one feature at whole.
                '''
                out = tf.keras.layers.Dropout(rate=self.dropout,
                                              noise_shape=[tf.shape(out)[0], 1])(out, training=self.is_training, )
            return out

    def safe_ids_for_emb(self, ids, voc_size):
        """
        if id >= voc_size, then it set to 0 which means UNK, and the embedding
        should be all zeros.
        """
        return tf.where(tf.less_equal(ids, voc_size),
                        ids, tf.zeros_like(ids))

    def sparse_embedding(self, ids, layer_name, voc_size, emb_size,
                         initial_range, name_scope='embedding',
                         combiner='mean'):
        with tf.name_scope(name_scope):
            emb = self.init_embedding_weights(
                layer_name, voc_size, emb_size, initial_range, name_scope)
            out = tf.nn.embedding_lookup_sparse(emb, ids, None,
                                                combiner=combiner)
            return out

    def build_sparse_embedding(self, ids, layer_name, voc_size,
                               emb_size):
        out = self.sparse_embedding(
            ids, layer_name, voc_size, emb_size, None,
            combiner=self.flags.sparse_embedding_combiner)
        return out
