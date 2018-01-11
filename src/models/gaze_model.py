import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.models.base_model import BaseModel

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
            self.loss = self.mse()
            self.optimize = self.optimizer(config=config)

    @base_utils.config_checker(['dropout_keep_prob',
                                'num_conv_layers_1',
                                'num_feature_1',
                                'kernel_1',
                                'max_pool_1',
                                'num_conv_layers_2',
                                'num_feature_2',
                                'kernel_2',
                                'max_pool_2',
                                'num_fc_layers',
                                'fc_layer_num'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            # TODO: Add batch normalization
            # TODO: More residual skip connections
            # TODO: Larger number of filters per layers
            # tf.summary.image('input_image', x)
            for _ in range(config.num_conv_layers_1):
                x = slim.conv2d(x, config.num_feature_1, kernel_size=config.kernel_1)
            if config.max_pool_1:
                x = slim.max_pool2d(x, [2, 2], scope='pool1')
            for _ in range(config.num_conv_layers_2):
                x = slim.conv2d(x, config.num_feature_2, kernel_size=config.kernel_2)
            if config.max_pool_2:
                x = slim.max_pool2d(x, [2, 2], scope='pool2')
            x = slim.flatten(x)
            x = slim.dropout(x, config.dropout_keep_prob, is_training=self.train_mode)
            for _ in range(config.num_fc_layers):
                x = slim.fully_connected(x, config.fc_layer_num)
                x = slim.dropout(x, config.dropout_keep_prob, is_training=self.train_mode)
            x = slim.fully_connected(x, 2, activation_fn=None)
        return x

    def mse(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict)
            tf.summary.scalar('mse', mse)
        return mse
