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
The refiner network 'refines' a synthetic image, making it more real.
'''


class RefinerModel(BaseModel):

    @base_utils.config_checker(['refiner_learning_rate',
                                'refiner_optimizer_type'])
    def __init__(self, config=None):
        super().__init__(config=config)
        self.label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
        # Reassign optimizer parameters
        config.learning_rate = config.refiner_learning_rate
        config.optimizer_type = config.refiner_optimizer_type
        self.build_graph(config=config)

    @base_utils.config_checker(['refiner_initializer',
                                'image_channels'])
    def model_func(self, config=None):
        with tf.variable_scope('refiner_model', initializer=config.refiner_initializer, reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            x = layers.resnet(x, self, config=config)
            x = slim.conv2d(x, config.image_channels, [1, 1], scope='final_refiner_conv')
            self.add_summary('output_image', x, 'image')
        return x

    @base_utils.config_checker(['regularization_lambda'])
    def loss_func(self, config=None):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # All images are fake (synthetic), so labels are all 0
            labels = tf.zeros(shape=[tf.shape(self.label)[0]], dtype=tf.uint8)
            one_hot_labels = tf.one_hot(labels, 2)
            loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=one_hot_labels,
                                                        logits=self.label,
                                                        scope='loss_real')
            loss_reg = tf.losses.absolute_difference(predictions=self.image,
                                                     labels=self.predict,
                                                     weights=config.regularization_lambda,
                                                     scope='loss_reg')
            loss = tf.add(loss_real, loss_reg)
            self.add_summary('loss_real', loss_real, 'scalar')
            self.add_summary('loss_reg', loss_reg, 'scalar')
            self.add_summary('loss', loss, 'scalar')
        return loss
