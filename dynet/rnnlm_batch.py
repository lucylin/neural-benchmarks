from collections import Counter, defaultdict
from itertools import count, chain
import random
import argparse
import dynet as dy
import numpy as np
import sys

random.seed(100)

p = argparse.ArgumentParser()
p.add_argument("--data_path", metavar="PATH",
               help="""path where the data is stored as a Pandas DataFrame""")
p.add_argument("--test", metavar="PATH",
               help="""tests a model loaded from a specified path \
(uses gold standard inputs)""")
p.add_argument("--train", metavar="PATH",
               help="trains a model and saves it to a specified path")

### Hyper parameters ish
p.add_argument("--learning_rate", metavar="LR",type=float,default=0.001,
               help="""Learning rate for AdamOptimizer.""")
p.add_argument("-s","--hidden_size", dest="hidden_size", type=int,default=256,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=50,
               help="""Batch size to pass through the graph.""")
p.add_argument("-m","--max_epoch", dest="max_epoch", type=int,default=50,
               help="""Number of passes through the training data""")
p.add_argument("--dynet-mem", help="""Sets dynet memory; ignore in code""")

args = p.parse_args()
MB_SIZE=args.batch_size
MAX_EPOCH=args.max_epoch
START=1
STOP=2
UNK=0

# Read data file into list (generator) of sentences
def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with file(fname) as fh:
        for line in fh:
            sent = map(int, line.strip().split())
            yield sent


data=list(read(args.data_path))
nwords = max(chain(*data))+1

# Create model
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, 64))

# Word-level LSTM (layers=1, input=????, output=hidden_size, model)
RNN = dy.LSTMBuilder(1, 64, int(args.hidden_size), model)

# Softmax weights/biases on top of LSTM outputs
# (i.e., unembedding forward layer)
W_sm = model.add_parameters((nwords, 128))
b_sm = model.add_parameters(nwords)


# Build the language model graph
def calc_lm_loss(sents):
    dy.renew_cg()
    # parameters -> expressions
    W_exp = dy.parameter(W_sm)
    b_exp = dy.parameter(b_sm)

    # initialize the RNN
    f_init = RNN.initial_state()

    # get the wids and masks for each step
    tot_words = 0
    wids = []
    masks = []
    for i in range(len(sents[0])):
        wids.append([(sent[i] if len(sent)>i else STOP) 
                     for sent in sents])
        mask = [(1 if len(sent)>i else 0) for sent in sents]
        masks.append(mask)
        tot_words += sum(mask)
    
    # start the rnn by inputting START
    init_ids = wids[0]
    s = f_init.add_input(dy.lookup_batch(WORDS_LOOKUP,init_ids))

    # feed word vectors into the RNN and predict the next word    
    losses = []
    for i, (wid, mask) in enumerate(zip(wids, masks)[1:]):
        # calculate the softmax and loss      
        score = W_exp * s.output() + b_exp
        loss = dy.pickneglogsoftmax_batch(score, wid)
        # mask the loss if at least one sentence is shorter
        if mask[-1] != 1:
            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr, (1,), MB_SIZE)
            loss = loss * mask_expr
        losses.append(loss)
        # update the state of the RNN        
        wemb = dy.lookup_batch(WORDS_LOOKUP, wid)
        s = s.add_input(wemb) 
    
    return dy.sum_batches(dy.esum(losses)), tot_words


def train(modelpath, data):
    num_tagged = cum_loss = 0

    # Sort training sentences in descending order and count minibatches        
    data.sort(key=lambda x: -len(x))
    sent_order = [x*MB_SIZE for x in range(len(data)/MB_SIZE)]

    # Perform training
    for epoch in xrange(MAX_EPOCH):
        random.shuffle(sent_order)
        for i,sid in enumerate(sent_order,1):
            print "\rBatch {}/{}".format(i, len(sent_order)), 
            if i % (500/MB_SIZE) == 0:
                trainer.status()
                print cum_loss / num_tagged
                num_tagged = cum_loss = 0
            # train on the minibatch
            batch = data[sid:sid+MB_SIZE]
            print "size: {}x{}".format(len(batch), len(batch[0])),
            sys.stdout.flush()
            loss_exp, mb_words = calc_lm_loss(batch)
            cum_loss += loss_exp.scalar_value()
            num_tagged += mb_words
            loss_exp.backward()
            trainer.update()
        print "epoch %r finished" % epoch
        trainer.update_epoch(1.0)
    print "Done training model; saving to {}".format(modelpath)
    model.save(modelpath)


def test(modelpath, data):
    model.from_file(modelpath)
    # Sort training sentences in descending order and count minibatches        
    data.sort(key=lambda x: -len(x))
    sent_order = [x*MB_SIZE for x in range(len(data)/MB_SIZE)]

    dev_loss = dev_words = 0
    for sid in sent_order:
        loss_exp, mb_words = calc_lm_loss(data[sid:sid+MB_SIZE])
        dev_loss += loss_exp.scalar_value()
        dev_words += mb_words
    print "Loss:", dev_loss / dev_words

if args.train:
    train(args.train, data)
if args.test:
    test(args.test, data)

