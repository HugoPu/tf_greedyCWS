import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.contrib.eager.python import tfe

from config import FLAGS, get_config
from input import CWS_Input
from model2 import CWS_Model

def main(_):
    tfe.enable_eager_execution()

    with tfe.restore_variables_on_create(tf.train.latest_checkpoint('./save')):
        with tf.device(None):

            pku_config = get_config('pku')
            train_input = CWS_Input(pku_config)
            initializer = tf.random_normal_initializer(0.0, 1.0)
            with tf.variable_scope('model', initializer=initializer):
                model = CWS_Model(pku_config, train_input)
                lr = tfe.Variable(pku_config.lr, name="learning_rate")

            test_config = get_config('')
            test_input = CWS_Input(test_config, train_input)

            for i in range(pku_config.max_epochs):
                start_time = time.time()

                model.train(lr)

                lr.assign(lr / (1 + pku_config.edecay * (i + 1)))

                end_time = time.time()

                print 'Trained %s epoch(s) took %.lfs per epoch' % (
                i + 1, (end_time - start_time) / (i + 1))
                model.predict(test_input, test_config.output_path+'%d' % (i + 1))
                os.system('python score.py %s %s %d %d' % (test_config.data_path, test_config.output_path, i + 1, i + 1))
                # cws.save('epoch%d' % (eidx + 1))
                # print 'Current model saved'


if __name__ == '__main__':
    # Run the program with an optional function 'main'
    tf.app.run()