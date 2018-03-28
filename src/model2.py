import numpy as np
import tensorflow as tf

from collections import namedtuple

Sentence = namedtuple('Sentence',['score','score_expr','LSTMState','y','prevState','wlen','golden'])

class Embedding(tf.layers.Layer):
    """An Embedding layer."""
    def __init__(self, vocab_size, embedding_dim, embedding_name, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_name = embedding_name

    def build(self):
        self.embedding = self.add_variable(
            self.embedding_name+'_Embedding',
            shape=[self.vocab_size, self.embedding_dim],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True)

    def call(self, x):
      return tf.nn.embedding_lookup(self.embedding, x)

class FNN(tf.keras.Model):
    def __init__(self, input_dims, output_dims, layer_name):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layer_name = layer_name

    def build(self):
        self.W = self.add_variable(
            self.layer_name + '_W',
            shape=[self.input_dims],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True)
        self.b = self.add_variable(
            self.layer_name + '_b',
            shape=[self.input_dims],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True)

    def call(self, input, activation):
        return activation(tf.nn.xw_plus_b(input, self.W, self.b))

class Prediction(tf.keras.Model):
    def __init__(self, config):
        self.nhiddens = config.nhiddens
        self.word_dims = config.word_dims
        self.cell = tf.contrib.rnn.BasicLSTMCell(
            self.nhiddens,
            forget_bias=0.0,
            state_is_tuple=True,
        )
        self.prediction_FNN = FNN(config.nhiddens, config.word_dims, 'Prediction')

    def build(self):
        self.word_vector = self.add_variable(
            '<Bos>',
            shape=[self.word_dims],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True)

    def call(self, state=None):
        if state==None:
            state = self.cell.zero_state(1, tf.float32)
        y, new_state = self.cell(self.word_vector, state)
        prediction = self.prediction_FNN(y, tf.nn.tanh)
        return prediction, new_state

class Word(tf.keras.Model):
    def __init__(self, config, input):
        self.dropout_rate = config.dropout_rate
        self.char_dims = config.char_dims
        self.word_dims = config.word_dims

        self.word_vocab_size = len(input.word_dict)
        self.char_vocab_size = len(input.char_dict)
        self.word_dict = input.word_dict

        self.char_emb_lookup = Embedding(self.char_vocab_size, config.char_dims, 'Char')
        self.word_emb_lookup = Embedding(self.word_vocab_size, config.word_dims, 'Word')

        self.word_FNN = FNN(config.char_dims, config.word_dims, 'Com')
        self.reset_gate_FNN = FNN(config.char_dims, config.word_dims, 'Reset_Gate')

    def call(self, word_char_idx):
        word_char_embedding = self.char_emb_lookup(word_char_idx)
        word_char_emb_drop = tf.nn.dropout(word_char_embedding, self.dropout_rate)
        input_concat = tf.concat(word_char_emb_drop, 0)
        reset_gate = self.reset_gate_FNN(input_concat, tf.nn.sigmoid)
        reset_word = tf.multiply(reset_gate, input_concat)
        word = self.word_FNN(reset_word, tf.nn.tanh)
        if tuple(word_char_idx) in self.word_dict:
            id = self.word_dict[tuple(word_char_idx)]
            word = (word + tf.nn.word_emb_lookup(id)) / 2

        return word

class Score(tf.keras.Model):
    def __init__(self, config):
        self.word_dims = config.word_dims

    def build(self):
        self.score_U = self.add_variable(
            'score_U',
            shape=[self.word_dims],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True)
    def call(self, prediction, word, margin):
        return tf.multiply(prediction + self.score_U, word) + margin

class Greedy_Search(tf.keras.Model):
    def __init__(self, config, input):
        self._scope = None
        self.max_word_len = config.max_word_len
        self.is_training = config.is_training
        self.margin_loss_discount = config.margin_loss_discount
        self.prediction = Prediction(config)
        self.word = Word(config, input)
        self.score = Score(config)


    def call(self, sent_char_idx, sent_char_label):
        init_prediction, init_state = self.prediction()
        init_score = tf.constant(0.)
        # An object only have properties
        init_sentence = Sentence(score=init_score, score_expr=init_score, LSTMState=init_state,
                                 prediction=init_prediction, prevState=None, wlen=None, golden=True)

        start_agenda = init_sentence
        agenda = [start_agenda]

        now = None
        golden_sent = None

        for idx, _ in enumerate(sent_char_idx, 1):  # from left to right, character by character
            now = None
            # Loop all segmentaion, and record the highest score segmentation and golden segmentation
            for wlen in xrange(1, min(idx, self.max_word_len) + 1):
                word_char_idx = sent_char_idx[idx - wlen:idx]
                word = self.word(word_char_idx)

                # If the segementation looks like this, get the last sentence which last segmentataion is in idx-wlen
                sent = agenda[idx - wlen]

                if self.is_training:
                    # If last time and now separate correctly
                    golden = sent.golden and sent_char_label[idx - 1] == wlen

                    # Max-margin, if golden thne 0, separeted even not golden, add score
                    margin = self.margin_loss_discount * wlen if sent_char_label[idx - 1] != wlen else 0.

                    # Final score = max_margin_core + before_score + word_score + sentence_smoothness_score
                    # Do separating if the segmentation looks ok
                    score = self.score(sent.prediction, word, margin)
                else:
                    golden = False
                    score = self.score(sent.prediction, word)

                # Even the separation isn't correct, the score will bigger
                good = (now is None or now.score < score)
                if golden or good:
                    new_prediction, new_state = self.prediction(word, sent.LSTMState)
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


class CWS_Model(tf.keras.Model):
    def __init__(self, config, input):

        self.is_training = config.is_training
        self.shuffle_data = config.shuffle_data
        self.batch_size = config.batch_size
        self.momentum = config.momentum


        self.sents_char_idx = input.sents_char_idx
        self.sents_char_label = input.sents_char_label

        self._lr = config.lr
        self._optimizer = tf.train.MomentumOptimizer(self._lr, config.momentum)

        self.greedy_search = Greedy_Search(config, input)

    def train(self):

        n = len(self.sents_char_idx)
        idx_list = range(n)
        # Random the sentences
        if self.shuffle_data:
            np.random.shuffle(idx_list)

        nsamples = 0

        self._prediction_idx = []
        loss = []

        for idx in idx_list:
            agender, golden_sent = self.greedy_search(self.sents_char_idx[idx], label=self.sents_char_label[idx])

            # Get segmentation
            line_idx = []
            now = agender[-1]
            left = 0

            while now.prevState is not None:
                line_idx.append(self.sents_char_idx[idx][left:left+now.wlen])
                left += now.wlen
                now = now.prevState

            self._prediction_idx.append(line_idx)

            if self.is_training:
                loss.append(agender[-1].score_expr - golden_sent.score_expr)

                if nsamples % self.batch_size == 0:
                    self._optimizer.apply_gradients(self._cost(loss))


    def _cost(self, loss):
        return tf.reduce_sum(loss)