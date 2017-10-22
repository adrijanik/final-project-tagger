# -*- coding: utf-8 -*-
from data_processor import DataProcessor
from disambiguer import Disambiguer
from lxml import objectify, etree
import networkx as nx


def disambiguation_generator():
    with open('tagged_graph_update1626.txt') as f:
        for line in f:
            line = line.split(" ")
            yield line[0],line[1]


def tag_dataset_for_graph(filename='../data/test-analyzed.xml'):
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    disambiguer = Disambiguer(filename)
    scope = 0
    setted = False
    lexems = []
    ctags = disambiguer.get_ctags_graph()
    disambs = disambiguation_generator()
    disambig = disambs.next()
    previous_orth = disambig[0].strip()
    for n, i in enumerate(ctags):
        lexem = i.getparent()
        lexems.append(lexem)
        orth = lexem.getparent().orth.text
        if orth != previous_orth:
            if not setted:
                ctags[n-1].getparent().set('disamb', str(int(True)))
            scope = 0
            setted = False
            disambig =  disambs.next()
            previous_orth = disambig[0].strip()
        else:
            scope += 1

        if disambig[1].strip() == i: 
            lexem.set('disamb', str(int(True)))
            setted = True
        elif len(lexem.getparent().lex) == 1:
            lexem.set('disamb', str(int(True)))
            setted = True
        elif disambig[1].strip() == "NotAvailable" and scope == 0:
            lexem.set('disamb', str(int(True)))
            setted = True

    docinfo = disambiguer.tree.docinfo
    out = etree.tostring(disambiguer.root, pretty_print=True, encoding=docinfo.encoding, xml_declaration=True)
    with open('TEST_ALL.xml', 'wb') as f:
        f.write(out)
   
    return lexems

def analyse_sentence(sentence, j, disamb, processor):
#    print("------------------------------SENTENCE nr {}: {}----------------------".format(j,sentence))
#    try:
    G = nx.DiGraph(nx.drawing.nx_pydot.read_dot('test_sentences/new_test_sentence'+str(j)+'.dot'))
    for ed in G.edges():
#        G[ed[0]][ed[1]][0]['weight'] = int(G[ed[0]][ed[1]][0]['weight'])
        G[ed[0]][ed[1]]['weight'] = int(G[ed[0]][ed[1]]['weight'])

#    print(G.edges(data=True))

    solid_node_indexes = []
    for i in range(len(sentence)):
  #      print(processor.words[sentence[i]][0])
        if len(processor.words[sentence[i]][0]) == 1:
            solid_node_indexes.append(i)
    if 0 not in solid_node_indexes:
        solid_node_indexes = [-1] + solid_node_indexes
        for k in processor.words[sentence[0]][0]:
            G.add_node('fake_start')
            G.add_edge('fake_start',k[1],weight=0)
    if len(sentence)-1 not in solid_node_indexes:
        solid_node_indexes.append(-2)
        for k in processor.words[sentence[len(sentence)-1]][0]:
            G.add_node('fake_end')
            G.add_edge(k[1],'fake_end',weight=0)
#    print(G.neighbors('fake_start'))
#    print("Solid state nodes: " + " ; ".join(str(el) for el in solid_node_indexes))
    paths = []
    if len(sentence) == 1:
        disamb.append((sentence[0],processor.words[sentence[0]][0][0][1]))
    else:
        for i in range(len(solid_node_indexes)-1):
            ind = solid_node_indexes[i]
            ind1 = solid_node_indexes[i+1]
 
            if ind == -1:

                if ind1 == -2:
                    shortest_paths = nx.shortest_simple_paths(G,'fake_start', 'fake_end')
                    length = len(sentence) + 2
                else:
                    length = ind1 + 2
                    shortest_paths = nx.shortest_simple_paths(G,'fake_start', processor.words[sentence[ind1]][0][0][1])
                pth = []
                  
                cnt = 0
                breaking = False
                for p in shortest_paths:
                    if length < len(p):
                        length = len(p)
 
                    if len(p) == length:
                        pth = p
                        break
                    if cnt >= 100:
                        pth = p
                        breaking = True
                        break
                    cnt += 1
 

 
                if pth == []:
                    pth = p * length
 
                if breaking:
                    pth = ['NotAvailable'] * length
 
                paths = paths + pth[1:]
 
            else:
                shortest_paths = nx.shortest_simple_paths(G, processor.words[sentence[ind]][0][0][1],processor.words[sentence[ind1]][0][0][1])
                if ind1 == -2:
                    length = (len(sentence) - 1) - ind + 1 

                else:
                    length = ind1 - ind + 1
 
                pth = []
                cnt = 0
                breaking = False
                for p in shortest_paths:
                    if length < len(p):
                        length = len(p)


                    if len(p) == length:
                        pth = p
                        break
                    if cnt >= 1000:
                        pth = p
                        breaking = True
                        break
                    cnt += 1
                if pth == []:
                    pth = p * length
                    
                if breaking:
                    pth = ['NotAvailable'] * length
                if ind == 0:
                    paths = paths + pth
                else:
                    paths = paths + pth[1:] 
        for i in range(len(sentence)):
            disamb.append((sentence[i],paths[i]))


def analyse_sentence_graph_dijkstra(filename='../data/test-analyzed.xml'):
    processor = DataProcessor(filename)
    processor.create_words_dictionary(gold=False)
    
    j = 0
    disamb = []
    for sentence in processor.sentences:
        analyse_sentence(sentence, j, disamb, processor)

        if j % 500 == 0:
            with open('tagged_graph' + str(j) + '.txt', 'wb') as f:
                f.write('\n'.join(["{} {}".format(d[0],d[1])  for d in disamb]))

        j += 1
    with open('tagged_graph_update' + str(j) + '.txt', 'wb') as f:
        f.write('\n'.join(["{} {}".format(d[0],d[1])  for d in disamb]))

