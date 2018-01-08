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
'''


class RefinerModel(BaseModel):

    @base_utils.config_checker(['refiner_learning_rate',
                                'refiner_optimizer_type'])
    def __init__(self, config=None):
        # Reassign optimizer parameters
        config.learning_rate = config.refiner_learning_rate
        config.optimizer_type = config.refiner_optimizer_type
        super().__init__(config=config)
        self.label = tf.placeholder(tf.int32, shape=(None,), name='true_real_or_fake')
        self.pred = tf.placeholder(tf.float32, shape=(None,), name='pred_real_or_fake')
        with tf.variable_scope('discriminator_model'):
            self.predict = self.model()
            self.optimize = self.optimizer(config=config)

    @base_utils.config_checker(['refiner_initializer'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=config.refiner_initializer,
                               reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
        return x

    @base_utils.config_checker(['regularization_lambda'])
    def loss(self, config=None):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.add(self.loss_realism,
                          tf.scalar_mul(config.regularization_lambda,
                                        self.loss_regularization))
            tf.summary.scalar('loss', loss)
        return loss

    def loss_realism(self):
        with tf.variable_scope('loss_realism', reuse=tf.AUTO_REUSE):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred,
                                                           labels=self.label)
            tf.summary.scalar('loss_realism', loss)
        return loss

    def loss_regularization(self):
        with tf.variable_scope('loss_regularization', reuse=tf.AUTO_REUSE):
            loss = tf.losses.cosine_distance(predictions=self.image,
                                             labels=self.predict)
            tf.summary.scalar('loss_regularization', loss)
        return loss
