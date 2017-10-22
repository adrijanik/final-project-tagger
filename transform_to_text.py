from disambiguer import Disambiguer

def transform_to_text(filename):
    disamb = Disambiguer(filename)
    ctags = disamb.get_gold_ctags()
    with open('ctags_as_text.txt', 'w') as f:
        f.write(' '.join([str(ctag) for ctag in ctags]))

