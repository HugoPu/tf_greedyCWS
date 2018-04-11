import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from collections import namedtuple

Sentence = namedtuple('Sentence',['score','score_expr','LSTMState','prediction','prevState','wlen','golden'])

class Prediction(tf.layers.Layer):
    def __init__(self, config):
        super(Prediction, self).__init__()

        self.nhiddens = config.nhiddens
        self.word_dims = config.word_dims
        self.cell = tf.contrib.rnn.BasicLSTMCell(
            self.nhiddens,
            forget_bias=0.0,
            state_is_tuple=True,
        )
        self.prediction_FNN = tf.keras.layers.Dense(
            units=config.word_dims,
            activation=tf.nn.tanh,
            input_shape=(config.nhiddens,),
            name='Prediction')

    def call(self, inputs, state=None):
        if state==None:
            state = self.cell.zero_state(1, tf.float32)
        y, new_state = self.cell(inputs, state)
        prediction = self.prediction_FNN(y)
        return prediction, new_state

class Word(tf.layers.Layer):
    def __init__(self, config, input):
        super(Word, self).__init__()

        char_dims = config.char_dims
        word_dims = config.word_dims
        word_vocab_size = len(input.word_dict)
        char_vocab_size = len(input.char_dict)

        self.dropout_rate = config.dropout_rate
        self.word_dict = input.word_dict

        char_init = tf.random_uniform_initializer(-0.5/char_dims, 0.5/char_dims)
        self.char_emb_lookup = tf.keras.layers.Embedding(
            input_dim=char_vocab_size,
            output_dim=char_dims,
            embeddings_initializer=char_init,
            name='Char')

        self.word_emb_lookup = tf.keras.layers.Embedding(
            input_dim=word_vocab_size,
            output_dim=word_dims,
            name='Word')

        self.word_FNN = []
        self.reset_gate_FNN = []
        for i in range(1, config.max_word_len+1):
            self.word_FNN.append(
                tf.keras.layers.Dense(
                    units=config.word_dims,
                    input_shape=(config.char_dims*i,),
                    activation=tf.nn.tanh,
                    name='Com_' + str(i)))
            self.reset_gate_FNN.append(
                tf.keras.layers.Dense(
                    units=config.char_dims*i,
                    input_shape=(config.char_dims*i,),
                    activation=tf.nn.sigmoid,
                    name='Reset_Gate_' + str(i)))

    def call(self, inputs):
        wlen = inputs.shape[0]
        word_char_embedding = self.char_emb_lookup(inputs)
        word_char_emb_drop = tf.nn.dropout(word_char_embedding, self.dropout_rate)
        input_concat = tf.reshape(word_char_emb_drop, [1, -1])
        reset_gate = self.reset_gate_FNN[wlen-1](input_concat)
        reset_word = tf.multiply(reset_gate, input_concat)
        word = self.word_FNN[wlen-1](reset_word)
        word_tuple = tuple(inputs.numpy())
        if word_tuple in self.word_dict:
            id = self.word_dict[word_tuple]
            word = (word + self.word_emb_lookup(tf.constant(id))) / 2

        return word

class Greedy_Search(tf.layers.Layer):
    def __init__(self, config, input):
        super(Greedy_Search, self).__init__()

        self.word_dims = config.word_dims
        self.max_word_len = config.max_word_len
        self.margin_loss_discount = config.margin_loss_discount

        self.prediction = Prediction(config)
        self.word = Word(config, input)

    def build(self, _):
        self.init_word = self.add_variable(
            'Bos',
            shape=[1, self.word_dims],
            dtype=tf.float32,
            trainable=True)

        self.score_U = self.add_variable(
            'score_U',
            shape=[self.word_dims],
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs, label=None):
        init_prediction, init_state = self.prediction(self.init_word)
        init_score = tf.constant(0.)
        # An object only have properties
        init_sentence = Sentence(score=init_score.numpy(), score_expr=init_score, LSTMState=init_state,
                                 prediction=init_prediction, prevState=None, wlen=None, golden=True)

        agenda = [init_sentence]

        now = None
        golden_sent = None

        for idx, _ in enumerate(inputs, 1):  # from left to right, character by character
            now = None
            # Loop all segmentaion, and record the highest score segmentation and golden segmentation
            for wlen in xrange(1, min(idx, self.max_word_len) + 1):
                word_char_idx = inputs[idx - wlen:idx]
                word = self.word(word_char_idx)

                # If the segementation looks like this, get the last sentence which last segmentataion is in idx-wlen
                sent = agenda[idx - wlen]

                if label is not None:
                    now_golden = label[idx - 1].numpy() == wlen
                    # If last time and now separate correctly
                    golden = sent.golden and now_golden

                    # Max-margin, if golden thne 0, separeted even not golden, add score
                    margin = self.margin_loss_discount * wlen if not now_golden else 0.

                    # Final score = max_margin_core + before_score + word_score + sentence_smoothness_score
                    # Do separating if the segmentation looks ok
                    score = tf.reduce_sum(tf.multiply(sent.prediction + self.score_U, word)) + margin
                else:
                    golden = False
                    score = tf.reduce_sum(tf.multiply(sent.prediction + self.score_U, word))

                # Even the separation isn't correct, the score will bigger
                good = (now is None or now.score < score.numpy())
                if golden or good:
                    new_prediction, new_state = self.prediction(word, state=sent.LSTMState)
                    # Update Sentence state
                    new_sent = Sentence(score=score.numpy(), score_expr=score, LSTMState=new_state,
                                        prediction=new_prediction, prevState=sent, wlen=wlen, golden=golden)
                    if good:
                        now = new_sent
                    if golden:
                        golden_sent = new_sent
            # Record the highest score segmentation at this word
            agenda.append(now)
            # If the golden score is now the highest one
            wrong_sep = label is not None and label[idx - 1].numpy() > 0 and (not now.golden)
            if wrong_sep:
                break
        return agenda, golden_sent


class CWS_Model(tfe.Network):
    def __init__(self, config, input):

        super(CWS_Model, self).__init__()

        self.shuffle_data = config.shuffle_data
        self.batch_size = config.batch_size
        self.momentum = config.momentum

        self.inputs = input.sents_char_idx
        self.labels = input.sents_char_label

        self.greedy_search = Greedy_Search(config, input)

    def predict(self, inputs, output_path):

        sents_char_idx = inputs.sents_char_idx
        lines = inputs.lines

        def seg(sent_char_idx, sentence):
            lens = []
            agender, _ = self.greedy_search(tf.constant(sent_char_idx),None)
            # Get segmentation
            now = agender[-1]
            while now.prevState is not None:
                lens.append(now.wlen)
                now = now.prevState
            lens.reverse()

            res, begin = [], 0
            # Separate the sentence based on the word lenght list
            for wlen in lens:
                res.append(''.join(sentence[begin:begin + wlen]))
                begin += wlen
            return res

        fo = open(output_path, 'wb')
        seq_idx = 0
        for line in lines:
            sent = line.split()  # character list
            Left = 0
            output_sent = []
            for idx, word in enumerate(sent):
                if len(re.sub('\W', '', word, flags=re.U)) == 0:
                    if idx > Left:
                        words = seg(sents_char_idx[seq_idx], list(''.join(sent[Left:idx])))
                        seq_idx += 1
                        output_sent.extend(words)
                    Left = idx + 1
                    output_sent.append(word)
            if Left != len(sent):
                words = seg(sents_char_idx[seq_idx], list(''.join(sent[Left:])))
                seq_idx += 1
                output_sent.extend(words)
            output_sent = '  '.join(output_sent).encode('utf8') + '\r\n'
            fo.write(output_sent)
        fo.close()

    def train(self, lr):
        # Random the sentences
        if self.shuffle_data:
            np.random.shuffle(self.inputs)

        num_batch = len(self.inputs) // self.batch_size

        for i in xrange(num_batch):
            inputs = self.inputs[i * self.batch_size:(i + 1) * self.batch_size]
            labels = self.labels[i * self.batch_size:(i + 1) * self.batch_size]
            grads = tfe.implicit_gradients(self._cost)(inputs, labels)
            tf.train.MomentumOptimizer(lr, self.momentum).apply_gradients(grads)

    def _cost(self, inputs, labels):
        result = []
        for i in xrange(len(inputs)):
            agenda, golden_sent = self.greedy_search(tf.constant(inputs[i]), tf.constant(labels[i]))
            result.append(agenda[-1].score_expr - golden_sent.score_expr)
            # print agenda[-1].score - golden_sent.score

        return result