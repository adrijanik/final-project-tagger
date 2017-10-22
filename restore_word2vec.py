# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import networkx as nx
import collections
from collections import Counter
import sys
import os
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from matplotlib import pylab
import matplotlib.pyplot as plt

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
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def restore_model():
    filename = './ctags_as_text.txt'
    
    words = read_data(filename)
    cnt = Counter(words)
    print(len(cnt))
    print('Data size %d' % len(words))
    
    vocabulary_size = len(cnt)#1331
    
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    
    with tf.Session() as sess:    
        lst = os.listdir('./embeddings_log')
        pth = ''
        for item in lst:
            if 'meta' in item:
                pth = item
        saver = tf.train.import_meta_graph('./embeddings_log/'+ pth)
        saver.restore(sess,tf.train.latest_checkpoint('./embeddings_log/'))
        graph = tf.get_default_graph()
        vocabulary_size = len(cnt)#1331
    
        valid_examples = np.array(range(vocabulary_size))
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
        embeddings = graph.get_tensor_by_name("word_embedding:0")
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings),name='similarity')
        sim = similarity.eval()
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[0, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % 'adj:sg:gen:m3:pos'
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)
        
        from data_processor import DataProcessor
        import networkx as nx
        import matplotlib.pylab as p
        
        filename = '../data/test-analyzed.xml'
        processor = DataProcessor(filename)
        processor.create_words_dictionary(gold=False)
        for form in processor.words['bo'][0]:
            print(form[1])
        k = 0
        for sentence in processor.sentences:
            G=nx.DiGraph()
            for i in range(len(sentence)-1):
                for form in processor.words[sentence[i]][0]:
                    for form1 in processor.words[sentence[i+1]][0]:
                        if form[1] in dictionary:
                            nearest = (-sim[dictionary[form[1]], :]).argsort()
                        if form1[1] in dictionary:
                            if dictionary[form1[1]] in nearest:
                                G.add_edge(form[1],form1[1], weight=list(nearest).index(dictionary[form1[1]]))
                        else:
                            G.add_edge(form[1],form1[1], weight=sys.maxint)
            if not os.path.exists('./test_sentences'):
                os.makedirs('./test_sentences')
            nx.drawing.nx_pydot.write_dot(G,'./test_sentences/test_sentence'+str(k)+'.dot')
            p.show()
            k += 1        
    
        final_embeddings = normalized_embeddings.eval()
    
        num_points = 30
        
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :]) 
    
        def plot(embeddings, labels):
          assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
          pylab.figure(figsize=(15,15))  # in inches
          for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
          plt.savefig('tsne.png') 
    
          plt.show()
        words = [reverse_dictionary[i] for i in range(1, num_points+1)]
        plot(two_d_embeddings, words) 
