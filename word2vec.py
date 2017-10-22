# Based on Deep models for text and sequences, Deep Learning by Google MOOC course Udacity 

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import sys
from collections import Counter

def read_data(filename):
    with open(filename) as f:
        data = f.read().split()
    return data

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, count, dictionary, reverse_dictionary

def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def train_model():
    filename = './ctags_as_text.txt'
     
    words = read_data(filename)
    cnt = Counter(words)
    print(len(cnt))
    print('Data size %d' % len(words))
    
    vocabulary_size = len(cnt)
    
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words
    
    data_index = 0
    
    
    print('data:', [reverse_dictionary[di] for di in data[:8]])
    
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        data_index = 0
        batch, labels = generate_batch(data, data_index, batch_size=8, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    
    
    tf.reset_default_graph()
    from tensorflow.contrib.tensorboard.plugins import projector
    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 2 
    valid_size = 16 
    valid_window = 100 
    valid_examples = np.array(range(vocabulary_size))
    num_sampled = 64 
    
    graph = tf.Graph()
    
    with graph.as_default(), tf.device('/cpu:0'):
    
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        LOG_DIR = './embeddings_log'
        embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='word_embedding')
        softmax_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                               stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        config = projector.ProjectorConfig()
      
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings.name
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
      
        summary_writer = tf.summary.FileWriter(LOG_DIR)
      
        projector.visualize_embeddings(summary_writer, config)
        
    
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        loss = tf.reduce_mean(
          tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                     labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
      
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    
    num_steps = len(data)/batch_size
    epochs = 30
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        labs = []
        for epoch in range(epochs):
            for step in range(num_steps):
                batch_data, batch_labels = generate_batch(
                  data, data_index, batch_size, num_skips, skip_window)
                feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                labs.append(batch_labels)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            if epoch % 5 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
          
        final_embeddings = normalized_embeddings.eval()
        saver = tf.train.Saver()
        saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)


