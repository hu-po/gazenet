import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.base_model import BaseModel

'''

The refiner network 'refines' a synthetic image, making it more real.

The input:
* A synthetic image

The output:
* The refined synthetic image

'''


class RefinerModel(BaseModel):

    @base_utils.config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        with tf.variable_scope('discriminator_model'):
            self.predict = self.predict_func()

    def predict_func(self):
        with tf.variable_scope('predict', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
        return x

    def refiner_loss(self):
        pass
