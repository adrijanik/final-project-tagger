import re
from collections import Counter
import pandas as pd
import numpy as np

### DATA PREPARATION ###
class DataProcessor():

    def __init__(self, filename):
        self.chunks = None
        self.sentences = []
        self.words = {}
        self.words_count = None
        self.data_tuples = None

        print("[read_training_data] Read in training chunks")
        with open(filename) as f:
            content = f.read()
            pattern = '<\?xml version="1\.0" encoding="UTF-8"\?>\s*<\!DOCTYPE cesAna SYSTEM "xcesAnaIPI\.dtd">\s*<cesAna xmlns\:xlink="http\:\/\/www\.w3\.org\/1999\/xlink" version="1\.0" type="lex disamb">\s*<chunkList>\s*(?P<chunks>[\W\s\d\w]+)<\/chunkList>\s*<\/cesAna>'
            chunks_block = re.search(pattern, content)
            if chunks_block:
                all_chunks = chunks_block.groups('chunks')
                pattern = '<chunk type=\"s\">\s*(?P<chunk>[.\w\W\s]+?)<\/chunk>\s*'
                self.chunks = re.findall(pattern, all_chunks[0])
    
    def create_words_dictionary(self, gold=True):
        print("[create_dictionary_train] Create dictionary from chunks")
        print("Number of chunks: {0}".format(len(self.chunks)))
        for chunk in self.chunks:
            pattern = '(?P<token><tok>\s*(?:[\w\W\d.]+?)<\/tok>\s*?)(?:<ns\/>)?'
            tokens = re.findall(pattern, chunk)
            sentence = []
            for tok in tokens:
                pattern = '<orth>(?P<orth>.+)<\/orth>\s*(?:[\w\W\d.]+)'
                orth = re.search(pattern, tok)
                x = orth.group('orth')
                sentence.append(x)
                if gold:
                    pattern = '<lex disamb=\"1\"><base>(?P<base>.+)<\/base><ctag>(?P<ctag>.+)<\/ctag><\/lex>\s*'
                    lexes = re.findall(pattern, tok)
                    self.words[x] = [lexes[0][1]]    
                else:
                    pattern = '<lex><base>(?P<base>.+)<\/base><ctag>(?P<ctag>.+)<\/ctag><\/lex>\s*'
                    lexes = re.findall(pattern, tok)
                    self.words[x] = [lexes]
            self.sentences.append(sentence)

def split_data_into_training_and_test_sets():
    ### SPLIT DATA INTO TRAINING AND TEST SETS ###
    correct = pd.DataFrame()
    non_correct = pd.DataFrame()
    correct_test = pd.DataFrame()
    non_correct_test = pd.DataFrame()
    
    for j, chunk in enumerate(pd.read_csv('input-output-dataset.csv', chunksize=10000)):
        del chunk['0']            
        del chunk['Unnamed: 0']
        
        if j % 5 == 0:
            correct_test = pd.concat([correct_test, chunk[chunk['disamb'] == True]])
            non_correct_test = pd.concat([non_correct_test, chunk[chunk['disamb'] == False]])
        else:
            correct = pd.concat([correct, chunk[chunk['disamb'] == True]])
            non_correct = pd.concat([non_correct, chunk[chunk['disamb'] == False]])
            bar_all.update(j)
    
    del correct['disamb']
    del non_correct['disamb']
    del correct_test['disamb']
    del non_correct_test['disamb']

   
    with open('in-out_correct.csv', 'w') as f:
        correct.to_csv(f, header=False, index=False)
    with open('in-out_non-correct.csv', 'w') as f:
        non_correct.to_csv(f, header=False, index=False)
    with open('in-out_correct_test.csv', 'w') as f:
        correct_test.to_csv(f, header=False, index=False)
    with open('in-out_non-correct_test.csv', 'w') as f:
        non_correct_test.to_csv(f, header=False, index=False)



