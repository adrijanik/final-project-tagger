import lxml.etree as ET
from lxml import objectify, etree
from sklearn.metrics import roc_curve, auc, roc_auc_score
from StringIO import StringIO
import matplotlib.pyplot as plt
import numpy as np

def get_sentences(root):
    chunks = []
    for j in root.chunkList.chunk:
        for ch in j.chunk:
            chunks.append(ch)

    return chunks


def xml_generator(path):
    xml = None
    with open(path, 'rb') as f:
        xml = f.read()

    tree = etree.parse(StringIO(xml))
    root = objectify.fromstring(xml)
    sentences = get_sentences(root)
    for sent in sentences:
        for tok in sent.tok:
            yield tok

def get_sentences_tag(root):
    chunks = []
    for j in root.chunkList.chunk:
        chunks.append(j)

    return chunks


def xml_generator_tag(path):
    xml = None
    with open(path, 'rb') as f:
        xml = f.read()

    tree = etree.parse(StringIO(xml))
    root = objectify.fromstring(xml)
    sentences = get_sentences_tag(root)
    for sent in sentences:
        for tok in sent.tok:
            yield tok

def evaluate():
    path_tagged = 'TEST_ALL.xml'
    tagged_gen = xml_generator_tag(path_tagged)
    
    path_gold = '../data/gold-task-a-b.xml'
    xml_generator(path_gold)
    gold_gen = xml_generator(path_gold)
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    All = 0
    cond = True
    predicted = []
    actual = []
    while cond:
        try:
            gold = gold_gen.next()
            tagged = tagged_gen.next()
        except:
            break
        for lex in gold.lex:
            for lex_tag in tagged.lex:
                if lex.ctag.text == lex_tag.ctag.text:
                    if 'disamb' in lex.attrib:
                        TP += 1
                        actual.append(1)
                        predicted.append(1)
                    else:
                        FP += 1
                        actual.append(0)
                        predicted.append(1)
                else:
                    if 'disamb' in lex.attrib:
                        FN += 1
                        actual.append(0)
                        predicted.append(0)
                    else:
                        TN += 1
                        actual.append(1)
                        predicted.append(0)
           
    print("All toks: {}, TP: {}, TN: {}, FN: {}, FP: {}".format(TP+TN+FN+FP,TP,TN,FN,FP))
    TP = float(TP)
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)
    accuracy = (TP + TN)/(TP+TN+FN+FP)
    precision = TP/(TP+FP) 
    recall = TP/(FN+TP)
    fscore = 2.0*(precision*recall)/(precision+recall)
    print("Results: Accuracy: {}, Precision: {}, Recall: {}, F-score: {}".format(accuracy, precision, recall, fscore))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(actual, predicted)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(actual.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
