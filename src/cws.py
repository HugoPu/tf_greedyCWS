import time
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.contrib.eager.python import tfe

from config import FLAGS, get_config
from input import CWS_Input
from model2 import CWS_Model

def main(_):
    # if not FLAGS.data_path:
    #     raise ValueError('Must set --data_path to PTB data directory')
    # gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    # if FLAGS.num_gpus > len(gpus):
    #     raise ValueError('Your machine has only %d gpus '
    #                      'which is less than the requested --num_gpus=%d.'
    #                      % (len(gpus), FLAGS.num_gpus))
    #
    # config = get_config()
    #
    # with tf.Graph().as_default():
    #     with tf.name_scope('Train'):
    #         train_input = CWS_Input(config)
    #         with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
    #             train_model = CWS_Model(train_input, config)
    #
    #     with tf.name_scope('Validate'):
    #         validate_input = CWS_Input(config)
    #         with tf.variable_scope('Model', reuse=True):
    #             validate_model = CWS_Model(validate_input, config)
    #
    #     with tf.name_scope('Test'):
    #         test_input = CWS_Input(config)
    #         with tf.variable_scope('Model', reuse=True):
    #             test_model = CWS_Model(test_input, config)
    #
    # with tf.Graph().as_default():
    #     init = tf.global_variables_initializer()
    #     with tf.Session() as sess:
    #         sess.run(init)
    #         for i in range(config.max_epochs):
    #             sess.run(train_model.lr_update,{train_model.new_lr:config.lr / (1 + config.edecay * (i+1))})
    #
    #             print('Epoch: %d Learning rate: %.3f' % (i + 1, sess.run(train_model.lr)))
    #             train_cost = run_epoch(sess, train_model, config)

    tfe.enable_eager_execution()

    with tfe.restore_variables_on_create(tf.train.latest_checkpoint('./save')):
        with tf.device(None):
            config = get_config()
            input = CWS_Input(config)
            model = CWS_Model(config, input)
            lr = tfe.Variable(config.lr, name="learning_rate")

            for i in range(config.max_epochs):
                model.train()
                lr.assign(lr/ (1 + config.edecay * (i + 1)))


if __name__ == '__main__':
    # Run the program with an optional function 'main'
    tf.app.run()