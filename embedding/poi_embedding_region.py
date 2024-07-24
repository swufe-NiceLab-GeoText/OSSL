#!/usr/bin/python3
# coding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.query_embedding = Embedding(vocab_size,
                                     embedding_dim,
                                     input_length=1,
                                     name="query_em_layer")
    self.poi_embedding = Embedding(vocab_size,
                                   embedding_dim,
                                   input_length=num_ns+1,
                                   name='poi_em_layer')
    self.dots = Dot(axes=(3, 1))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    word_emb0 = self.query_embedding(target[:, 0])
    word_emb1 = self.query_embedding(target[:, 1])
    word_emb = (word_emb0 + word_emb1)/2
    context_emb = self.poi_embedding(context)
    dots = self.dots([context_emb, word_emb])

    return self.flatten(dots)


class PoiEmbedding():

    def __init__(self, trajectories, poi_num, num_ns=3):
        self.poi_num = poi_num
        self.trajectories = trajectories
        self.num_ns = num_ns
        self.positive_data = []
        self.dataset = None


    def gen_train(self):
        for trajectory in trajectories:
            true = set()
            for poi in trajectory:
                true.add(poi)
            for i in range(1, len(trajectory) - 1):
                target = (trajectory[i - 1], trajectory[i + 1])
                self.positive_data.append((target, trajectory[i], list(true)))

        print('Postive sample:',len(self.positive_data))

        targets, contexts, labels = [], [], []
        for target_pois, context_poi, true_class in self.positive_data:
            context_class = tf.constant(np.array([true_class]), dtype=tf.int64)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=len(true_class),
                num_sampled=self.num_ns, # rate of positive:negitive is 1:self.num_ns
                unique=True,
                range_max=self.poi_num,
                seed=40,
                name="nagetive_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
            context_poi = tf.reshape(context_poi, [1,1])
            context_poi = tf.cast(context_poi,dtype=tf.int64)
            context = tf.concat([context_poi, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * self.num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_pois)         # destinetion  target tuple(start, end)
            contexts.append(context)            # one positive+num_ns*negtive   context Tensor shape(1+num_ns,1)
            labels.append(label)                # context label 1DTensor  shape(1+num_ns,)
        print(len(targets),len(contexts),len(labels))

        BATCH_SIZE = 256
        BUFFER_SIZE = 10000
        self.dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        self.dataset = self.dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    def train(self, em_size, name):
        embedding_dim = em_size   # poi embedding dim
        word2vec = Word2Vec(self.poi_num, embedding_dim, self.num_ns)
        word2vec.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        word2vec.fit(self.dataset, epochs=15, callbacks=[tensorboard_callback])

        que_weights = word2vec.get_layer('query_em_layer').get_weights()[0]
        poi_weights = word2vec.get_layer('poi_em_layer').get_weights()[0]
        que_weights = pd.DataFrame(que_weights)
        print(que_weights.shape)
        poi_weights = pd.DataFrame(poi_weights)
        print(poi_weights.shape)
        poi_weights.to_csv('./self-embedding/embedding{}.csv'.format(name), index=False)


poi_embedding_size = 128
if __name__ == '__main__':

    poi_num = 8059
    with open('./data/train_CDtrajectories_osr1.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    start_time = time.time()
    self_embedding = PoiEmbedding(trajectories, poi_num)
    self_embedding.gen_train()
    name = '_CD_osr1_region'
    # name = '_CD_osr2_region'
    # name = '_CD_osr3_region'
    self_embedding.train(poi_embedding_size, name)
    print(time.time() - start_time)

