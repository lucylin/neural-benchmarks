#!/usr/bin/env python3

import sys, os, re
from datetime import timedelta
import argparse

from model import *

VOCAB_SIZE = 42530 # words + OOV, doesnt include start or stop

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
               help="""Learning rate for AdamOptimizer.""")
p.add_argument("--hidden_size", dest="hidden_size", type=int,default=256,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=50,
               help="""Batch size to pass through the graph.""")
p.add_argument("--max_epoch", dest="max_epoch", type=int,default=150,
               help="""Determines how many passes throught the training data to do""")


if len(sys.argv) < 2:
  p.print_help()
  exit(2)
  
args = p.parse_args()
log(args)

def load_data(path):
  with open(path) as f:
    docs = [[int(w) for w in l.split(" ")] for l in f.readlines()[:30]]
  max_seq_len = max(map(len,docs))
  vocab_size = max(map(max,docs))
  #assert vocab_size == VOCAB_SIZE , "Vocab size mismatch "+\
  #  str(vocab_size)+" should be "+str(VOCAB_SIZE)
  
  return docs,max_seq_len,vocab_size

def train(model_path, data_path):
  docs,max_seq_len,vocab_size = load_data(data_path)
  args.max_seq_len = max_seq_len
  args.vocab_size = vocab_size
  
  m = LanguageModel(args)
  
  val_m = LanguageModel(args,train=False,
                        reuse=True,model=m)
  
  embed()

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

