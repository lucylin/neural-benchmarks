#!/usr/bin/env python3

import os, sys
from collections import Counter
import json
from IPython import embed
from numpy.random import shuffle

MAX_LENGTH = 100
LOWERCASE = True
VOCAB_CUTOFF = 20
DEFAULT_VOCAB_FILE = "/home/msap/data/LDC/tok/vocab.json"
DEFAULT_DIR = "/home/msap/data/LDC"

def make_vocab(docs,vocab_json=DEFAULT_VOCAB_FILE):
    if os.path.isfile(vocab_json):
        with open(vocab_json) as f:
            word_to_id = json.load(f)
    else:
        all_words = [w for l in docs for w in l]
        counts = Counter(all_words)
        words = [w for w,c in counts.items() if c >= VOCAB_CUTOFF]
        coverage = sum([c for w,c in counts.items() if w not in words])
        print("{:.4f} % of data get's OOVed with a cutoff of {}".format(
            100*coverage/len(all_words),VOCAB_CUTOFF))
        words.insert(0,'<EOS>')
        words.insert(0,'<BOS>')
        words.insert(0,'<OOV>')
        word_to_id = {w:i for i,w  in enumerate(words)}
        with open(vocab_json,"w+") as f:
            json.dump(word_to_id,f)
            print("Exported the word->id dict to the following JSON file:\n"+vocab_json)
    print("Vocab size (all words + OOV,EOS,BOS symbols) =",len(word_to_id))
    
    return word_to_id

def tok_docs(docs, w2id):
    return [[w2id['<BOS>']]+[w2id.get(w, w2id['<OOV>']) for w in d]+[w2id['<EOS>']] for d in docs]
    
def tok_dir(path):
    print("Reading in files")
    if LOWERCASE:
        all_lines = [l.strip().lower().split(" ") for f in os.listdir(path)
                     for l in open(os.path.join(path,f)).readlines() if f[-4:] == ".txt"]
    else:
        all_lines = [l.strip().split(" ") for f in os.listdir(path)
                     for l in open(os.path.join(path,f)).readlines() if f[-4:] == ".txt"]
    assert max(map(len,all_lines)) <= MAX_LENGTH, "ERROR"
    print("max(map(len,all_lines)) =",max(map(len,all_lines)))
    print("Read files.")

    w2id = make_vocab(all_lines)
    print("Made or loaded vocab")
    
    ids = tok_docs(all_lines,w2id)
    assert len(ids) == len(all_lines), "Something went wrong"
    print("Tokenized all documents; splitting into train/val set")

    lim = len(ids) // 10
    order = list(range(len(ids)))
    shuffle(order)
    val = [ids[i] for i in order[:lim]]
    train = [ids[i] for i in order[lim:]]
    
    assert len(train) + len(val) == len(ids), "Error splitting"
    print("done splitting")
    
    train_fn = os.path.join(DEFAULT_DIR,"tok/train.txt")
    with open(train_fn,"w+") as f:
        f.write("\n".join([" ".join(map(str,d)) for d in train]))
        f.write("\n")
    print("Wrote tokenized / integerized documents to",train_fn)
    val_fn = os.path.join(DEFAULT_DIR,"tok/val.txt")
    with open(val_fn,"w+") as f:
        f.write("\n".join([" ".join(map(str,d)) for d in val]))
        f.write("\n")
    print("Wrote tokenized / integerized documents to",val_fn)

if __name__=="__main__":
    tok_dir(sys.argv[1])
