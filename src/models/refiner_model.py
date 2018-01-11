import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.models.base_model import BaseModel

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
        self.label = tf.placeholder(tf.float32, shape=(None, 2), name='pred_label')
        with tf.variable_scope('refiner_model'):
            self.predict = self.model(config=config)
            self.loss_reg = self.loss_regularization(config=config)
            self.loss_real = self.loss_realism()
            self.loss = self.combined_loss()
            self.optimize = self.optimizer(config=config)

    @base_utils.config_checker(['refiner_initializer',
                                'num_resnet_blocks',
                                'num_feat_per_resnet_block',
                                'kernel_size_resnet_block'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=config.refiner_initializer,
                               reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')

            # Quick function that implements
            def resnet_block(input, num_features, kernel_size):
                x = slim.conv2d(input, num_features, kernel_size, padding='same')
                output = tf.concat([x, input], axis=3)
                return output

            for _ in range(config.num_resnet_blocks):
                x = resnet_block(x, config.num_feat_per_resnet_block, config.kernel_size_resnet_block)
            x = slim.conv2d(x, config.image_channels, [1, 1], scope='final_resnet_conv')
            self.add_summary('output_image', x, 'image')
        return x

    def combined_loss(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.add(self.loss_real, self.loss_reg)
            self.add_summary('loss', loss)
        return loss

    def loss_realism(self):
        with tf.variable_scope('loss_realism', reuse=tf.AUTO_REUSE):
            # All images are fake, so labels are all 0
            labels = tf.zeros(shape=[tf.shape(self.label)[0]], dtype=tf.uint8)
            one_hot_labels = tf.one_hot(labels, 2)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=one_hot_labels,
                                                   logits=self.label)
            self.add_summary('loss_realism', loss)
        return loss

    @base_utils.config_checker(['regularization_lambda'])
    def loss_regularization(self, config=None):
        with tf.variable_scope('loss_regularization', reuse=tf.AUTO_REUSE):
            loss = tf.losses.absolute_difference(predictions=self.image,
                                                 labels=self.predict,
                                                 weights=config.regularization_lambda)
            self.add_summary('loss_regularization', loss)
        return loss
