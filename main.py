from analyse_sentences import *
from evaluation import *
from restore_word2vec import *
from transform_to_text import *
from word2vec import *
from process_sentences import *
from tagger import *

print("----------STAGE 1: DATA PREPARATION----------")
transform_to_text('../data/train-gold.xml')
print("----------STAGE 2: TRAINING MODEL----------")
train_model()
print("----------STAGE 3: ANALYSING TEST DATA----------")
restore_model()
process_sentences()
print("----------STAGE 4: GRAPH CREATION----------")
analyse_sentence_graph_dijkstra()
path = '../data/test-analyzed.xml'
disamb_path = 'tagged_graph_update1626.txt'
tag_file(path, disamb_path)
print("----------STAGE 5: FINAL TAGGING----------")
tag_dataset_for_graph()
print("----------STAGE 6: EVALUATION----------")
evaluate()
