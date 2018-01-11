import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.models.base_model import BaseModel
import src.models.custom_layers as layers

'''
This network outputs the gaze location for a given webcam image.
This model is inspired by the architechture in [1].
'''


class GazeModel(BaseModel):

    @base_utils.config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        with tf.variable_scope('input'):
            self.label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
        self.build_graph(config=config)

    @base_utils.config_checker()
    def model_func(self, config=None):
        with tf.variable_scope('model', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            # Model arch is a stack of conv blocks with residual connections, then a fully connected head
            x = layers.resnet(x, self, config=config)
            x = layers.dim_reductor(x, self, config=config)
            x = layers.fc_head(x, self, config=config)
            # Final layer for regression has no activation function
            x = slim.fully_connected(x, 2, activation_fn=None, scope='output')
        return x

    def loss_func(self, config=None):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.debug = tf.Print(self.label, [self.label, self.predict])
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict, scope='mse')
            self.add_summary('mse', mse, 'scalar')
        return mse
