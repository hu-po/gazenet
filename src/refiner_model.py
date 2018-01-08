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
            self.predict = self.model()

    @base_utils.config_checker(['refiner_initializer'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=config.refiner_initializer,
                               reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
        return x

    @base_utils.config_checker(['regularization_lambda'])
    def loss(self, config=None):
        loss = tf.add(self.loss_realism,
                      tf.scalar_mul(config.regularization_lambda,
                                    self.loss_regularization))

    def loss_realism(self):
        pass

    def loss_regularization(self):
        pass
