import itertools
import logging
import numpy as np
import time
import defaults

from keras.utils.np_utils import to_categorical
from nn import RNNLM

UNK = 0
BOS = 1
EOS = 2

VOCAB_SIZE = 35642+1 # words + OOV + BOS + EOS + 1 to correct for UNK!=0

class LanguageModel:
    '''
    Basic language model using Keras-based LSTM implementation.
    '''

    def __init__(self):
        self._model = None
        self._log = logging.getLogger(self.__class__.__name__)

    def configure_logger(self, level=logging.DEBUG, write_file=False):
        fmt = '%(asctime)s %(name)s.%(funcName)-20s %(levelname)-8s \
               %(message)s'

        kwargs = {
            'level': level,
            'format': fmt,
            'datefmt': '%m-%d %H:%M',
        }

        if write_file:
            log_path = 'theano.log'  # TODO: better.

            kwargs['filename'] = log_path
            kwargs['filemode'] = 'w'

        logging.basicConfig(**kwargs)
        self._log = logging.getLogger(self.__class__.__name__)

    def _load_sequences(self, data_path):
        self._log.info('Loading data from {0}...'.format(data_path))

        with open(data_path, 'r') as f:
            seqs = [[int(x) for x in line.strip().split()] for line in f]

        # +1 since 0-indexed
        vocab_size = max(max(s) for s in seqs) + 1

        self._log.debug('Data loaded!')
        return seqs, vocab_size

    '''Training-specific functions.'''

    def train(self, train_path, output_path=None,
              learning_rate=defaults.LEARNING_RATE,
              hidden_size=defaults.HIDDEN_SIZE, batch_size=defaults.BATCH_SIZE,
              max_epoch=defaults.MAX_EPOCH):

        self._log.info('Enter: train({0})'.format(locals()))

        new_unk = VOCAB_SIZE + 1

        def get_data(data_path):
            seqs, vocab_size = self._load_sequences(data_path)
            max_seq_len = max(len(s) for s in seqs)

            if vocab_size > VOCAB_SIZE:
                raise Exception('unexpectedly large vocab size')

            return self._format_data(seqs, new_unk), max_seq_len

        (data, labels), max_seq_len = get_data(train_path)
        self._log.info('Shapes: {0}\t{1}'.format(data.shape, labels.shape))

        # yay, hackery
        val_path = train_path.replace('train', 'val')
        (val_data, val_labels), max_seq_len_val = get_data(val_path)
        self._log.info('Shapes: {0}\t{1}'.format(val_data.shape, val_labels.shape))

        n_val_seqs = val_data.shape[0]
        if max_seq_len_val > max_seq_len:
            # for sanity -- truncate this to max_seq_len
            val_data = val_data[:, :max_seq_len]
            val_labels = val_labels[:, :max_seq_len, :]
        elif max_seq_len_val < max_seq_len:
            val_data = np.concatenate((
                val_data, np.zeros((n_val_seqs, max_seq_len-max_seq_len_val))
            ), axis=1)
            val_labels = np.concatenate((
                val_labels, np.zeros((n_val_seqs, max_seq_len-max_seq_len_val))
                ), axis=1)

        # TODO: not this
        #val_data = data[:batch_size,:]
        #val_labels = labels[:batch_size,:,:]
        #data = data[batch_size:]
        #labels = labels[batch_size:]

        self._log.info('Initializing LSTM parameters/functions...')
        rnnlm = RNNLM(hidden_size, new_unk+1, batch_size, learning_rate,
                      max_seq_len - 1)  # -1 for pred
        self._log.debug('Done initializing LSTM parameters/functions.')

        self._log.info(rnnlm.model.summary())
        n_batches = data.shape[0] // batch_size
        n_val_batches = val_data.shape[0] // batch_size

        loss_history = []

        try:
            for epoch in range(max_epoch):
                self._log.info('Starting epoch {0}'.format(epoch))
                start_time = time.time()
                for batch in range(n_batches):
                # (TODO) for profiling...
                #for batch in range(1):  # just train on 1 batch
                    # need to transform
                    idx = batch_size * batch
                    batch_labels = expand_labels(labels[idx:idx+batch_size,:], new_unk+1)
                    h = rnnlm.train_batch(data[idx:idx+batch_size,:],
                                          batch_labels)
                    self._log.info('Batch {0} info: {1}'.format(batch,
                                    h.history))

                rnnlm.model.reset_states()

                # (TODO) for profiling...
                # just validate on 1 batch
                '''
                val_labels2 = expand_labels(val_labels[:batch_size,:], new_unk+1)
                val_loss = rnnlm.validate(val_data[:batch_size,:],
                                          val_labels2)
                '''
                total_loss = 0.
                for batch in range(n_val_batches):
                    idx = batch_size * batch
                    val_batch_labels = expand_labels(val_labels[idx:idx+batch_size,:], new_unk+1)
                    val_loss = rnnlm.validate(val_data[idx:idx+batch_size,:],
                                              val_batch_labels)
                    self._log.info('Cross-entropy loss: {0}'.format(val_loss))
                    total_loss += val_loss[1]
                    rnnlm.model.reset_states()

                loss_history.append(total_loss / n_val_batches)

                self._log.info('Avg loss: {0}'.format(total_loss/n_val_batches))
                self._log.info('Finished epoch {0} in {1} s'.format(
                                epoch, time.time() - start_time))

                # is the loss increasing?
                if (len(loss_history) > 2 and
                    loss_history[-1] > loss_history[-2] and
                    loss_history[-2] > loss_history[-3]):
                    break

                rnnlm.model.reset_states()

        except KeyboardInterrupt:
            self._log.warn('Training interrupted')

        '''
        if output_path is not None:
            self._log.info('Saving model to {}...'.format(output_path))
            self.save_model(output_path)
            self._log.debug('Model successfully saved!')
        '''

        self._log.debug('Exit: train()')

    def _format_data(self, seqs, new_unk):
        max_seq_len = max(len(s) for s in seqs)
        n_seqs = len(seqs)
        data = np.zeros((n_seqs, max_seq_len), dtype=np.int32)

        for i, seq in enumerate(seqs):
            for t, w in enumerate(seq):
                w_fixed = w if w != UNK else new_unk
                data[i, t] = w_fixed

        return data[:,:-1], data[:,1:]

    '''Test-specific functions.'''

    '''
    def predict(self, model_path, data_path, output_path=None):
        self._log.debug('Enter: predict({0})'.format(locals()))

        if not self._model:
            raise AttributeError('Model not loaded')

        self._log.debug('Exit: predict()')
    '''

def expand_labels(labels, max_label):
    n_seqs, max_seq_len = labels.shape
    expanded = np.zeros((n_seqs, max_seq_len, max_label), dtype=np.int32)
    for s in range(n_seqs):
        for t in range(max_seq_len):
            w = labels[s,t]
            expanded[s,t,w] = 1

    return expanded
