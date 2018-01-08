import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.base_model import BaseModel

'''

The discriminator network is trained to differentiate between synthetic and real
images.

The input:
* A real or synthetic image

The output:
* Single softmax output, denoting either real or fake image

'''


class DiscriminatorModel(BaseModel):

    @base_utils.config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        with tf.variable_scope('discriminator_model'):
            self.predict = self.predict_func()

    def predict_func(self):
        with tf.variable_scope('predict', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
            x = slim.conv2d(x, 96, [3, 3], stride=2, scope='conv1')
            x = slim.conv2d(x, 64, [3, 3], stride=2, scope='conv2')
            x = slim.max_pool2d(x, [3, 3], scope='pool1')
            x = slim.conv2d(x, 32, [3, 3], scope='conv3')
            x = slim.conv2d(x, 32, [1, 1], scope='conv4')
            x = slim.conv2d(x, 2, [1, 1], scope='conv5')
            x = slim.flatten(x)
            x = slim.softmax(x)
        return x
