import lxml.etree as ET
from lxml import objectify, etree

from StringIO import StringIO

def get_sentences(root):
    chunks = []
    for j in root.chunkList.chunk:
            chunks.append(j)
    return chunks


def disambiguation_generator(file_name):
    with open(file_name) as f:
        for line in f:
            line = line.split(" ")
            yield line[0],line[1]


def xml_generator(path, file_name):
    xml = None
    with open(path, 'rb') as f:
        xml = f.read()

    tree = etree.parse(StringIO(xml))
    root = objectify.fromstring(xml)
    sentences = get_sentences(root)
    for sent in sentences:
        for tok in sent.tok:
            yield tok

    ET.ElementTree(root).write(file_name, encoding="UTF-8", xml_declaration=True, pretty_print=True)


def tag_file(path, disamb_path):
    gen = xml_generator(path,'TAGGED.xml')
    disambs = disambiguation_generator(disamb_path)
    
    for i in gen:
        disamb = disambs.next()[1].strip()
        setted = False
        for j,k in enumerate(i.lex):
            if k.ctag.text == disamb and setted == False:
                k.set('disamb',str(int(True)))
                setted = True
            if j == len(i.lex)-1 and setted == False:
                 k.set('disamb',str(int(True)))


