#!/usr/bin/env python3

import sys, os, re
from datetime import timedelta
import argparse
import pickle
import numpy as np
from numpy.random import shuffle
from model import *

VOCAB_SIZE = 35642 # words + OOV + BOS + EOS

p = argparse.ArgumentParser()
p.add_argument("--data_path", metavar="PATH",
               help="""path where the data is stored as a Pandas DataFrame""")
p.add_argument("--test", metavar="PATH",
               help="""tests a model loaded from a specified path (uses gold standard inputs)
               """)
p.add_argument("--train", metavar="PATH",
               help="trains a model and saves it to a specified path")

### Hyper parameters ish
p.add_argument("--learning_rate", metavar="LR",type=float,default=0.001,
               help="""Learning rate for AdamOptimizer. (Default 0.001)""")
p.add_argument("--hidden_size", dest="hidden_size", type=int,default=256,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=50,
               help="""Batch size to pass through the graph.""")
p.add_argument("--max_epoch", dest="max_epoch", type=int,default=50,
               help="""Determines how many passes throught the training data to do""")


if len(sys.argv) < 2:
  p.print_help()
  exit(2)
  
args = p.parse_args()
log(args)

def load_data(path):
  pkl = path.replace(".txt",".pkl")    
  
  if os.path.isfile(pkl):
    with open(pkl,"rb") as f:
      docs = pickle.load(f)
      assert len(docs) == 1191848, pkl
  else:
    with open(path) as f:
      docs = [[int(w) for w in l.split(" ")] for l in f.readlines()]
    with open(pkl,"wb+") as f:
      pickle.dump(docs,f)
  
  max_seq_len = max(map(len,docs))
  vocab_size = max(map(max,docs)) + 1 # cause 0 indexed
  assert vocab_size == VOCAB_SIZE , "Vocab size mismatch "+\
    str(vocab_size)+" should be "+str(VOCAB_SIZE)
  
  return docs,max_seq_len,vocab_size

def yield_batches(docs,args):
  """Yields `args.batch_size`-sized batches, padding with EOS tokens."""
  bs = args.batch_size
  n_yields = int(np.ceil(len(docs)/bs))
  log("Yielding {} batches".format(n_yields))
  
  eos = docs[0][-1] # presumably EOS
  assert all([s[-1] == eos for s in docs])
  
  for i in range(n_yields):
    chunk = docs[i*bs:(i+1)*bs]
    b_size = len(chunk)
    
    lens = np.array([len(s) for s in chunk])
    assert lens.shape == (b_size,)
    
    seqs = np.array([
      s+[eos]*(args.max_seq_len-len(s))
      for s in chunk])
    assert seqs.shape == (b_size,args.max_seq_len)
    
    yield seqs, lens


def train(model_path, data_path):
  """Trains the language model, testing on a validation set every """
  docs,max_seq_len,vocab_size = load_data(data_path)
  args.max_seq_len = max_seq_len
  args.vocab_size = vocab_size

  # splitting docs into training, validation (10% val)
  lim = len(docs)//10
  order = list(range(len(docs)))
  shuffle(order)
  val_docs = [docs[i] for i in order[:lim]]
  train_docs = [docs[i] for i in order[lim:]]
  
  # Getting batches
  val_b = [b for b in yield_batches(val_docs,args)]
  train_b = [b for b in yield_batches(train_docs,args)]

  # Building models/graphs
  m = LanguageModel(args)
  
  val_m = LanguageModel(args,train=False,
                        reuse=True,model=m)

  cost_history = []
  
  save_model(args.train,m)
  
  for step in range(args.max_epoch):
    log("Epoch {}".format(step+1))
    cost = m.train_epoch(train_b)
    log("Epoch {}: training cost = {:.4f}".format(step+1,cost))
    cost = val_m.train_epoch(val_b,cost_only=True)
    log("Epoch {}: validation cost = {:.4f}".format(step+1,cost))
    
    cost_history.append(cost)

    if len(cost_history) > 2 and cost_history[-1] > cost_history[-2] \
       and cost_history[-2] > cost_history[-3]:
      # overfitting?
      log("Overfitting, last four costs: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
        *cost_history[-4:]))
      save_model(args.train,m)
      break
    

def main(args):
  if args.train and args.data_path:
    train(args.train, args.data_path)
  elif args.test and args.reader_path:
    test(args.test, args.reader_path)
  elif args.plot and args.reader_path:
    plot(args.plot, args.reader_path)
  else:
    print("Specify --train")
    p.print_help()
    exit(2)


if __name__ == "__main__":
  start_time = time.time()
  main(args)
  time_d = time.time() - start_time
  log("Done! [That took {}]".format(timedelta(seconds=time_d)))

