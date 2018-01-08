import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.base_model import BaseModel

'''
This network outputs the gaze location for a given webcam image.
This model is inspired by the architechture in [1].

The input:
* Webcam image (128x98x3)
* Target gaze location (2x1)

The output:
* A trained refiner network
'''


class GazeModel(BaseModel):

    @base_utils.config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        self.label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
        self.train_mode = tf.placeholder(tf.bool, shape=[], name='train_mode_switch')
        with tf.variable_scope('gaze_model'):
            self.predict = self.model(config=config)
            self.loss = self.mse_func()
            self.optimize = self.optimizer(config=config)
            self.train_loss = self.train_loss_func()
            self.test_loss = self.test_loss_func()

    @base_utils.config_checker(['dropout_keep_prob'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
            x = slim.conv2d(x, 32, [3, 3], scope='conv1')
            x = slim.conv2d(x, 32, [3, 3], scope='conv2')
            x = slim.conv2d(x, 64, [3, 3], scope='conv3')
            x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')
            x = slim.conv2d(x, 64, [3, 3], scope='conv4')
            x = slim.conv2d(x, 128, [3, 3], scope='conv5')
            x = slim.max_pool2d(x, [2, 2], scope='pool2')
            x = slim.flatten(x)
            x = slim.dropout(x, config.dropout_keep_prob,
                             is_training=self.train_mode, scope='dropout1')
            x = slim.fully_connected(x, 256, scope='fc2')
            x = slim.dropout(x, config.dropout_keep_prob,
                             is_training=self.train_mode, scope='dropout2')
            x = slim.fully_connected(x, 2, activation_fn=None)
        return x

    def mse_func(self):
        with tf.variable_scope('mse', reuse=tf.AUTO_REUSE):
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict)
        return mse

    def train_loss_func(self):
        tf.summary.scalar('train_loss', self.loss)
        return self.loss

    def test_loss_func(self):
        tf.summary.scalar('test_loss', self.loss)
        return self.loss
