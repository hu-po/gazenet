import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.models.layers as layers
from src.models.model import Model

'''
This network outputs the gaze location for a given webcam image.
'''


class GazeModel(Model):

    def model_head(self, x):
        x = layers.dim_reductor(x, self)
        x = layers.fc_head(x, self)
        # Final layer for regression has no activation function
        x = slim.fully_connected(x, 2, activation_fn=None, scope='output')
        return x

    def loss_func(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict, scope='mse')
            self.add_summary('mse', mse, 'scalar')
        return mse
