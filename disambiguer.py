# -*- coding: utf-8 -*-
from lxml import objectify, etree
import pandas as pd
import numpy as np
import random
from StringIO import StringIO

def get_random():
    return random.choice([True, False])


### TAG INPUT FILE WITH CORRECT DISAMBIGUATIONS ###
class Disambiguer():
    def __init__(self, path):
        xml = None
        with open(path, 'rb') as f:
            xml = f.read()

        self.tree = etree.parse(StringIO(xml))
        self.root = objectify.fromstring(xml)
#        self.output = pd.read_csv('output_tags.csv').values

    def get_sentences(self):
        self.sentences = []
        self.chunks = []
        for j in self.root.chunkList.chunk:
            self.chunks.append(j.chunk)
            for i in j.chunk.tok:
                self.sentences.append(i.orth.text)
        print(' '.join(self.sentences))

        return self.chunks

    def get_ctags_graph(self):
        self.ctags = []
        for chunk in self.root.chunkList.chunk:
            for i in chunk.tok:
                for j in i.lex:
                    self.ctags.append(j.ctag)
        return self.ctags

    def get_ctags(self):
        self.ctags = []
        for chunk in self.root.chunkList.chunk:
            for i in chunk.tok:
                for j in i.lex:
                    self.ctags.append(j.ctag)
        return self.ctags

    def get_gold_ctags(self):
        self.ctags = []
        for chunk in self.root.chunkList.chunk:
            for i in chunk.chunk:
                for k in i.tok:
                    self.ctags.append(k.lex.ctag)
        return self.ctags

    def get_disambiguation(self, n, word, interpretation):
        #print(word + "  " + interpretation + " " + str(n))
        if n >= len(self.output):
            return get_random()
        if float(self.output[n][0]) >= float(self.output[n][1]):
            return True
        return False

    def tag_lexems_for_graph(self):
        lexems = []
        ctags = self.get_ctags()
        print(len(ctags))
        for n, i in enumerate(ctags):
            lexems.append(i.getparent())
            tmp = i.getparent().getparent().orth
            orth = tmp.text
            disamb = self.get_disambiguation_for_orth(orth)
            if disamb == True:
                i.getparent().set('disamb', str(int(disamb)))
        
        docinfo = self.tree.docinfo
        out = etree.tostring(self.root, pretty_print=True, encoding=docinfo.encoding, xml_declaration=True)
        with open('tagged_test_file.xml', 'wb') as f:
            f.write(out)

        return lexems

    def tag_lexems(self):
        lexems = []
        ctags = self.get_ctags()
        print(len(ctags))
        for n, i in enumerate(ctags):
            lexems.append(i.getparent())
            tmp = i.getparent().getparent().orth
            orth = tmp.text
            disamb = self.get_disambiguation(n, orth, i.text)
            if disamb == True:
                i.getparent().set('disamb', str(int(disamb)))
        
        docinfo = self.tree.docinfo
        out = etree.tostring(self.root, pretty_print=True, encoding=docinfo.encoding, xml_declaration=True)
        with open('tagged_test_file.xml', 'wb') as f:
            f.write(out)

        return lexems


