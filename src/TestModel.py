import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from config import FLAGS, get_config
from input import CWS_Input

tfe.enable_eager_execution()

class Embedding(tf.layers.Layer):
    """An Embedding layer."""
    def __init__(self, vocab_size, embedding_dim, embedding_name, **kwargs):
        super(Embedding, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_name = embedding_name

    def build(self, _):
        self.embedding = self.add_variable(
            self.embedding_name+'_Embedding',
            shape=[self.vocab_size, self.embedding_dim],
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs):
      return tf.nn.embedding_lookup(self.embedding, inputs)

class FNN(tf.layers.Layer):
    def __init__(self, input_dims, output_dims, layer_name):
        super(FNN, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layer_name = layer_name

    def build(self, _):
        self.W = self.add_variable(
            self.layer_name + '_W',
            shape=[self.input_dims, self.output_dims],
            dtype=tf.float32,
            trainable=True)
        self.b = self.add_variable(
            self.layer_name + '_b',
            shape=[self.output_dims],
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs, activation):
        return activation(tf.nn.xw_plus_b(inputs, self.W, self.b))

class Word(tf.layers.Layer):
    def __init__(self, config, input):
        super(Word, self).__init__()

        self.dropout_rate = config.dropout_rate
        self.char_dims = config.char_dims
        self.word_dims = config.word_dims

        self.word_vocab_size = len(input.word_dict)
        self.char_vocab_size = len(input.char_dict)
        self.word_dict = input.word_dict

        self.char_emb_lookup = Embedding(self.char_vocab_size, config.char_dims, 'Char')
        self.word_emb_lookup = Embedding(self.word_vocab_size, config.word_dims, 'Word')

        self.word_FNN = []
        self.reset_gate_FNN = []
        for i in range(1, config.max_word_len+1):
            self.word_FNN.append(FNN(config.char_dims*i, config.word_dims, 'Com_' + str(i)))
            self.reset_gate_FNN.append(
                FNN(config.char_dims*i, config.char_dims*i, 'Reset_Gate_' + str(i)))

    def call(self, inputs):
        wlen = inputs.shape[0]
        word_char_embedding = self.char_emb_lookup(inputs)
        word_char_emb_drop = tf.nn.dropout(word_char_embedding, self.dropout_rate)
        input_concat = tf.reshape(word_char_emb_drop, [1, -1])
        reset_gate = self.reset_gate_FNN[wlen-1](input_concat, activation=tf.nn.sigmoid)
        reset_word = tf.multiply(reset_gate, input_concat)
        word = self.word_FNN[wlen-1](reset_word, activation=tf.nn.tanh)
        word_tuple = tuple(inputs.numpy())
        if word_tuple in self.word_dict:
            id = self.word_dict[word_tuple]
            word = (word + self.word_emb_lookup(tf.constant(id))) / 2

        return word

config = get_config('pku')
input = CWS_Input(config)
model = Word(config, input)

def loss(model, x, y):
    y_ = model(x)
    return tf.reduce_sum(y - y_)

# inputs = [1263, 176, 664, 560, 605, 440, 1, 34, 177, 350]
for i in range(20):
    init = tf.constant_initializer(0.0)
    with tf.variable_scope('test',initializer=init):
        inputs = tf.constant([1263])
        embeddings = tf.ones((1, 50))
        grads = tfe.implicit_gradients(loss)(model, inputs, embeddings)
        tf.train.MomentumOptimizer(0.2, config.momentum).apply_gradients(grads)

    # print("Loss at step {:03d}: {:.3f}".format(i, loss(model, inputs, embeddings).numpy()))





