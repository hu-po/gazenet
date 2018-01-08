import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.base_model import BaseModel

'''
The discriminator network differentiates between synthetic and real images.
'''


class DiscriminatorModel(BaseModel):

    @base_utils.config_checker(['discriminator_learning_rate',
                                'discriminator_optimizer_type'])
    def __init__(self, config=None):
        # Reassign optimizer parameters
        config.learning_rate = config.discriminator_learning_rate
        config.optimizer_type = config.discriminator_optimizer_type
        super().__init__(config=config)  # Real image placeholder defined in base model
        self.label = tf.placeholder(tf.int32, shape=(None,), name='true_real_or_fake')
        self.pred = tf.placeholder(tf.float32, shape=(None,), name='pred_real_or_fake')
        with tf.variable_scope('discriminator_model'):
            self.predict = self.model(config=config)
            self.loss = self.discriminator_loss()
            self.optimize = self.optimizer(config=config)

    @base_utils.config_checker(['discriminator_initializer'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=config.discriminator_initializer,
                               reuse=tf.AUTO_REUSE):
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

    def discriminator_loss(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred,
                                                           labels=self.label)
            tf.summary.scalar('loss', loss)
        return loss
