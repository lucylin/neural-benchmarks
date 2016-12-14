import numpy as np
import theano
import theano.tensor as T

import defaults

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

class RNNLM:
    def __init__(self, hidden_size, vocab_size, batch_size, learning_rate,
                 max_seq_len):

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len

        self._build_graph()

    def _build_graph(self):
        model = Sequential()

        input_shape = (self.batch_size, self.max_seq_len)
        model.add(Embedding(self.vocab_size, self.hidden_size, mask_zero=True,
                  batch_input_shape=input_shape))

        model.add(LSTM(self.hidden_size, return_sequences=True, stateful=True,
                  consume_less='gpu'))   # change as needed

        # unemb/softmax layer
        model.add(TimeDistributed(Dense(self.vocab_size)))
        model.add(Activation('softmax'))

        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['categorical_crossentropy'])

        self.model = model

    def train_batch(self, data, labels):
        return self.model.fit(data, labels, nb_epoch=1,
                              batch_size=self.batch_size, shuffle=False)

    def validate(self, data, labels):
        return self.model.evaluate(data, labels, batch_size=self.batch_size)
