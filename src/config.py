import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'pku', "A type of model. Possible options are: small, medium, large")

flags.DEFINE_string('data_path', './simple-examples/data', 'Where the training/test data is stored.')

flags.DEFINE_string('save_path', './save_path', 'Model output directory.')

flags.DEFINE_bool('use_fp16', False, 'Train using 16-bit floats instead of 32bit float.')

flags.DEFINE_integer('num_gpus', 0, 'If larger than 1, Grappler AutoParallel optimizer '
                     'will create multiple training replicas with each GPU '
                     'running one replica.')

flags.DEFINE_string('rnn_mode', 'BASIC', 'The low level implementation of lstm cell: one of CUDNN, '
                    'BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, '
                    'and lstm_block_cell classes.')

FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'

class PKU_Config(object):
    max_epochs = 30
    batch_size = 256
    char_dims = 100
    word_dims = 50
    nhiddens = 51
    dropout_rate = 0.2
    max_word_len = 4
    load_params = None,  # None for train mode, otherwise please specify the parameter file.
    margin_loss_discount = 0.2
    max_sent_len = 60
    shuffle_data = True
    data_path = '../data/pku_train'
    threhold = 5.
    #dev_file = '../data/pku_test'  # dev/test in train/test mode.
    pre_trained = None
    lr = 0.2
    edecay = 0.2 # msr,pku 0.2,0.1
    momentum = 0.5
    word_proportion = 0.5  # we keep a short list H of the most frequent words
    is_training = True

class MSR_Config(object):
    max_epochs = 30
    batch_size = 256
    char_dims = 100
    word_dims = 50
    nhiddens = 50
    dropout_rate = 0.2
    max_word_len = 4
    load_params = None  # None for train mode, otherwise please specify the parameter file.
    margin_loss_discount = 0.2
    max_sent_len = 60
    shuffle_data = True
    data_path = '../data/pku_train'
    threhold = 5.
    #dev_file = '../data/pku_test'  # dev/test in train/test mode.
    pre_trained = None
    lr = 0.2
    edecay = 0.1  # msr,pku 0.2,0.1
    momentum = 0.5
    word_proportion = 0.5  # we keep a short list H of the most frequent words
    is_training = True

def get_config():
    config = None
    if FLAGS.model == 'pku':
        config = PKU_Config()
    else:
        config = MSR_Config()
    return config