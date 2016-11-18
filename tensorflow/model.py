#!/usr/bin/env python3

import os
import tensorflow as tf
from IPython import embed()


###############################################
######## Saving and loading functions #########
###############################################

def save_model(fn, model, ckpt=None):
  """Saves the TensorFlow variables to file"""
  if fn[-3] != ".tf":
    fn += ".tf"
  if model.saver is None:
    with model.graph.as_default():
      model.saver = tf.train.Saver()
  if ckpt is None:
    ckpt = fn.replace(".tf",".ckpt")
  ckpt = os.path.basename(ckpt)
  log("Saving model to {}".format(fn))
  model.saver.save(model.session, fn, latest_filename=ckpt)

def load_model(fn, model):
  """Loads the TensorFlow variables into the model (has to be constructed)"""
  if fn[-3] != ".tf":
    fn += ".tf"
  if model.saver is None:
    with model.graph.as_default():
      model.saver = tf.train.Saver()
  log("Loading model from {}".format(fn))
  model.saver.restore(model.session, fn)
  log("Done loading!")
  
###############################################

PRINT_FREQ = 100


class LanguageModel():
  """Simple RNN language model"""
  def __init__(args,train=True,reuse=None,model=None):
    """Builds the computation graph"""
    self.max_seq_len = args.max_seq_len
    self.vocab_size = args.vocab_size
    self.hidden_size = args.hidden_size

    initialize = model is None # whether to initialize variables

    # evice = "/cpu:0" if args.cpu else ""
    self.graph = tf.Graph() # if model is None else model.graph
    self.session = tf.Session(graph=self.graph) \
                   # if model is None else model.session

    with self.graph.as_default(),\
         tf.variable_scope("Language Model") as vs:
      self._seq = tf.placeholder(
        tf.int64,[None,max_seq_len])
      self._len = tf.placeholder(
        tf.int64,[None,])

    cell = tf.nn.rnn_cell.BasicLSTMCell(
      self.hidden_size,state_is_tuple=True)

    # Running RNN through sequence
    logit, _ = self.rnn_with_embedding(
      cell,None,self._seq, self._len,reuse=None)
    
    logit_list = tf.unpack(tf.transpose(logit,[1,0,2]))
    seq_list = tf.unpack(tf.transpose(self._seq,[1,0]))
    seq_list = seq_list[1:]

    xent = self.softmax_xent_loss_sequence(
      logit_list,seq_list,self._len,max_seq_len)
    
    self._cost = xent

    
    if train:
      log(vs.name+"/Adding optimizer")
      with tf.variable_scope("AdamOptimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self._train_op = optimizer.minimize(self._cost)
        
      if initialize:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.initialize_all_variables())
        
    log("Done with constructor.")

  def rnn_with_embedding(self,cell,init_state,input_seq,
                             input_seq_len,reuse=None,
                             scope="RNN"):
    """Given a sequence, embeds the symbols and runs it through a RNN.
    Returns:
      the unembedded outputs & final state at the right time step.

    Note: unembedded outputs are length-1 compared to input_seq_len!
    """    
    with tf.variable_scope(scope,reuse=reuse) as vs:
      log(vs.name+"/Encoding sequences")
      with tf.device('/cpu:0'):
        emb = tf.get_variable("emb",
                              [self.vocab_size,self.hidden_size],
                              dtype=tf.float32)
        un_emb = tf.get_variable("unemb",
                                 [self.hidden_size,self.vocab_size],
                                 tf.float32)
        # We need a bias
        un_emb_b = tf.get_variable("unemb_b",
                                   [self.vocab_size],
                                   dtype=tf.float32)
        
        assert scope+"/emb:0" in emb.name,\
          "Making sure the reusing is working"
        emb_input_seq = tf.nn.embedding_lookup(
          emb,input_seq)
        emb_input_list = tf.unpack(
          tf.transpose(emb_input_seq,[1,0,2]))
        
      # RNN pass
      if init_state is None:
        init_state = cell.zero_state(
          tf.shape(emb_input_list[0])[0],tf.float32)
        
      emb_output_list, final_state = tf.nn.rnn(
        cell,emb_input_list,initial_state=init_state,
        sequence_length=input_seq_len)

      # We shift the predicted outputs, because at
      # each word we're trying to predict the next.
      emb_output_list = emb_output_list[:-1]
      
      # Unembedding
      output_list = [tf.matmul(t,un_emb) + un_emb_b
                     for t in emb_output_list]
      outputs = tf.transpose(tf.pack(output_list),[1,0,2])

    return outputs, final_state

  def softmax_xent_loss_sequence(
      self,logits,target,seq_len,
      max_seq_len,reduce_mean=True):
    """Given a target sentence (and length) and
    un-normalized probabilities (logits) for predicted
    sentence, computes the cross entropy, excluding the
    padded symbols.
    Note: All arguments must be lists.
    """
    
    # Loss weights; don't wanna penalize PADs!
    ones = tf.ones_like(seq_len)
    ones_float = tf.ones_like(seq_len,dtype=tf.float32)
    zeros = ones_float*0
    weight_list = [
      tf.select(
        tf.less_equal(
          ones*i,seq_len-1),
        ones_float,zeros)
      for i in range(max_seq_len-1)]
    self._weights = tf.transpose(tf.pack(weight_list))
    
    # Loss function
    xent = tf.nn.seq2seq.sequence_loss_by_example(
      logits,target,weight_list)
    
    if reduce_mean:
      xent = tf.reduce_mean(xent)

    return xent
