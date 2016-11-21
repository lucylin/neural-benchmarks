#!/usr/bin/env python3

import sys, os, re
import time
from datetime import timedelta
import argparse

from model import *
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--data_path", metavar="PATH",
               help="""path where the data is stored as a Pandas DataFrame""")
p.add_argument("--reader_path", metavar="PATH",
               help="""path where the Reader object is stored as a pickle""")
# p.add_argument("--load", metavar="PATH",
#                help="""loads a model from specified path and tests it on the training data 
#                (at decoding, uses the true output as inputs to the RNN)""")
p.add_argument("--test", metavar="PATH",
               help="""tests a model loaded from a specified path (uses gold standard inputs)
               """)
p.add_argument("--plot", metavar="PATH",
               help="""plots the gradient from a model loaded from a specified path (uses gold standard inputs)
               run on a random sentence from the validation set.""")
p.add_argument("--train", metavar="PATH",
               help="trains a model and saves it to a specified path")
p.add_argument("--keep_training", metavar="PATH",
               help="loads a model/reader and keeps training & saving to the specified path")
#p.add_argument("--generate", metavar="PATH",
#               help="""loads a model from specified path and tests it on the training data 
#               (at decoding, samples at each step and uses that as input to the RNN""")

### Hyper parameters ish
p.add_argument("--learning_rate", metavar="LR",type=float,default=0.001,
               help="""Learning rate for AdamOptimizer.""")
p.add_argument("--reverse_prob",default=False,action="store_true",
               help="""Reverse the scoring prob from p(s|c) to p(c|s)""")
p.add_argument("--hidden_size", dest="hidden_size", type=int,default=256,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=50,
               help="""Batch size to pass through the graph.""")
p.add_argument("--max_epoch", dest="max_epoch", type=int,default=150,
               help="""Determines how many passes throught the training data to do""")
p.add_argument("--vocab_cutoff", dest="vocab_cutoff", type=int,default=3,
               help="""Determines cutoff for word frequency in training""")

p.add_argument("--no_overfit_safety", default=True, action='store_false',
               dest='overfit_safety',help="stop when validation cost stops decreasing")


if len(sys.argv) < 2:
  p.print_help()
  exit(2)
  
args = p.parse_args()
log(args)

def main(args):
  if args.train and args.data_path:
    train(args.train, args.data_path, args.reader_path)
  elif args.test and args.reader_path:
    test(args.test, args.reader_path)
  elif args.plot and args.reader_path:
    plot(args.plot, args.reader_path)
  else:
    print("Specify --train")
    p.print_help()
    exit(2)
