#!/usr/bin/env python3

import os, sys
from collections import Counter
import json
from IPython import embed

MAX_LENGTH = 100
LOWERCASE = True
VOCAB_CUTOFF = 150
DEFAULT_VOCAB_FILE = "/home/msap/data/LDC/preprocessed/vocab.json"


def make_vocab(docs,vocab_json=DEFAULT_VOCAB_FILE):
    if os.path.isfile(vocab_json):
        with open(vocab_json) as f:
            word_to_id = json.load(f)
    else:
        all_words = [w for l in docs for w in l]
        counts = Counter(all_words)
        words = [w for w,c in counts.items() if c >= VOCAB_CUTOFF]
        words.insert(0,'<OOV>')
        word_to_id = {w:i for i,w  in enumerate(words)}
        with open(vocab_json,"w+") as f:
            json.dump(word_to_id,f)
            print("Exported the word->id dict to the following JSON file:\n"+vocab_json)
    print("Vocab size (all words + OOV symbol) =",len(word_to_id))
    return word_to_id


def tok_docs(docs, w2id):
    return [[w2id.get(w, w2id['<OOV>']) for w in d] for d in docs]
    
def tok_dir(path):
    print("Reading in files")
    if LOWERCASE:
        all_lines = [l.strip().lower().split(" ") for f in os.listdir(path)
                     for l in open(os.path.join(path,f)).readlines()]
    else:
        all_lines = [l.strip().split(" ") for f in os.listdir(path)
                     for l in open(os.path.join(path,f)).readlines()]
    print("Read files.")
    w2id = make_vocab(all_lines)
    print("Made or loaded vocab")
    ids = tok_docs(all_lines,w2id)
    print("Tokenized all documents")
    with open(os.path.join(path,"tok_docs.txt"),"w+") as f:
        f.write("\n".join([" ".join(map(str,d)) for d in ids]))
        

if __name__=="__main__":
    tok_dir(sys.argv[1])
