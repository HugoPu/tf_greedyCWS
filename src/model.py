import numpy as np
import tensorflow as tf

from collections import namedtuple

Sentence = namedtuple('Sentence',['score','score_expr','LSTMState','prediction','prevState','wlen','golden'])

class CWS_Model(tf.keras.Model):
    def __init__(self, input, config):

        self.char_dims = config.char_dims
        self.word_dims = config.word_dims
        self.dropout_rate = config.dropout_rate
        self.max_word_len = config.max_word_len
        self.is_training = config.is_training
        self.margin_loss_discount = config.margin_loss_discount
        self.nhiddens = config.nhiddens
        self.shuffle_data = config.shuffle_data
        self.batch_size = config.batch_size
        self.momentum = config.momentum

        self.word_dict = input.word_dict
        self.char_dict = input.char_dict
        self.sents_char_idx = input.sents_char_idx
        self.sents_char_label = input.sents_char_label

        self.data_type = tf.float32

        self.global_steps = 0

        if self.is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
            self._lr_update = tf.assign(self._lr, self._new_lr)

        n = len(self.sents_char_idx)
        idx_list = range(n)
        # Random the sentences
        if self.shuffle_data:
            np.random.shuffle(idx_list)

        self._prediction_idx = []
        loss = []
        for idx in idx_list:
            agender, golden_sent = self._greed_search(self.sents_char_idx[idx], self.sents_char_label[idx])

            # Get segmentation
            line_idx = []
            now = agender[-1]
            left = 0
            while now.prevState is not None:
                line_idx.append(self.sents_char_idx[idx][left:left+now.wlen])
                left += now.wlen
                now = now.prevState

            self._prediction_idx.append(line_idx)

            if  self.is_training:
                loss.append(agender[-1].score_expr - golden_sent.score_expr)

                if self.global_steps % self.batch_size == 0:
                    self._cost = tf.reduce_sum(loss)
                    self._optimizer = tf.train.MomentumOptimizer(self._lr, self.momentum)
                    self._train_op = self._optimizer.minimize(self._cost,global_step=self.global_steps)


    def _greed_search(self,sent_char_idx, sent_char_label):
        init_prediction, init_state = self._sent_word_lstm()
        init_score = tf.constant(0.)
        # An object only have properties
        init_sentence = Sentence(score=init_score, score_expr=init_score, LSTMState=init_state,
                                 prediction=init_prediction,prevState=None, wlen=None, golden=True)

        start_agenda = init_sentence
        agenda = [start_agenda]

        now = None
        golden_sent = None

        for idx, _ in enumerate(sent_char_idx, 1):  # from left to right, character by character
            now = None
            # Loop all segmentaion, and record the highest score segmentation and golden segmentation
            for wlen in xrange(1, min(idx, self.max_word_len) + 1):
                word_char_idx = sent_char_idx[idx - wlen:idx]
                word = self._get_word_vector(word_char_idx)

                # If the segementation looks like this, get the last sentence which last segmentataion is in idx-wlen
                sent = agenda[idx - wlen]

                if self.is_training:
                    # If last time and now separate correctly
                    golden = sent.golden and sent_char_label[idx-1]==wlen

                    # Max-margin, if golden thne 0, separeted even not golden, add score
                    margin = self.margin_loss_discount * wlen if sent_char_label[idx - 1] != wlen else 0.

                    # Final score = max_margin_core + before_score + word_score + sentence_smoothness_score
                    # Do separating if the segmentation looks ok
                    score = self._get_score(sent.prediction, word, margin)
                else:
                    golden = False
                    score = self._get_score(sent.prediction, word)

                # Even the separation isn't correct, the score will bigger
                good = (now is None or now.score < score)
                if golden or good:
                    new_prediction, new_state = self._sent_word_lstm(word, sent.LSTMState)
                    # Update Sentence state
                    new_sent = Sentence(score=score, score_expr=score, LSTMState=new_state,
                                        prediction=new_prediction, prevState=sent, wlen=wlen, golden=golden)
                    if good:
                        now = new_sent
                    if golden:
                        golden_sent = new_sent
            # Record the highest score segmentation at this word
            agenda.append(now)
            # If the golden score is now the highest one
            if sent_char_label is not None and sent_char_label[idx - 1] > 0 and (not now.golden):
                break
        return agenda, golden_sent

    def _sent_word_lstm(self, word_vector=None, state=None):

        cell = tf.contrib.rnn.BasicLSTMCell(
            self.nhiddens,
            forget_bias=0.0,
            state_is_tuple=True,
        )

        if word_vector is None and state is None:
            state = cell.zero_state(1, self.data_type)
            word_vector = tf.get_variable('BoS', [1, self.word_dims], dtype=self.data_type)

        with tf.name_scope('LSTM'):
            y, new_state = cell(word_vector, state)
        with tf.name_scope('Prediction'):
            prediction_W = tf.get_variable('prediction_W', [self.nhiddens, self.word_dims], self.data_type)
            prediction_b = tf.get_variable('prediction_b', [self.word_dims], self.data_type)
            prediction = tf.tanh(tf.nn.xw_plus_b(y, prediction_W, prediction_b))
        return prediction, new_state

    def _get_score(self, prediction, word, margin=0):
        with tf.name_scope('Score'):
            score_U = tf.get_variable('word_score_U', [self.word_dims])
            return tf.multiply(prediction + score_U, word) + tf.constant(margin)

    def _get_word_vector(self, word_char_idx):
        wlen = len(word_char_idx)
        with tf.name_scope('Character_Embedding'):
            char_embeddings = tf.get_variable("char_embedding", [len(self.char_dict), self.char_dims],
                                              dtype=self.data_type)
            word_char_embedding = tf.nn.embedding_lookup(char_embeddings, word_char_idx)

        with tf.name_scope('Drop_Out'):
            word_char_emb_drop = tf.nn.dropout(word_char_embedding, self.dropout_rate)

        with tf.name_scope('Word_Inputs'):
            input_concat = tf.concat(word_char_emb_drop, 0)

        with tf.name_scope('Reset_Gate'):
            reset_gate_W = tf.get_variable('reset_gate_W' + str(wlen), [self.char_dims, self.char_dims],
                                           self.data_type)
            reset_gate_b = tf.get_variable('reset_gate_b' + str(wlen), [self.char_dims],
                                           self.data_type)
            reset_gate = tf.nn.sigmoid(tf.nn.xw_plus_b(input_concat, reset_gate_W, reset_gate_b))

        with tf.name_scope('Word'):
            com_W = tf.get_variable('com_W', [self.char_dims, self.word_dims], dtype=self.data_type)
            com_b = tf.get_variable('com_b', [self.word_dims], dtype=self.data_type)
            word = tf.nn.tanh(tf.nn.xw_plus_b(tf.multiply(reset_gate, input_concat), com_W, com_b))
            if tuple(word_char_idx) in self.word_dict:
                with tf.name_scope('Word_Embedding'):
                    word_embeddings = tf.get_variable("word_embedding", [len(self.word_dict), self.word_dims],
                                                      dtype=self.data_type)
                    id = self.word_dict[tuple(word_char_idx)]
                word = (word + tf.nn.embedding_lookup(word_embeddings, id)) / 2

        return word

    @property
    def prediction(self):
        return self._prediction_idx

    @property
    def new_lr(self):
        return self._new_lr

    @property
    def lr_update(self):
        return self._lr_update

    @property
    def train_op(self):
        return self._train_op