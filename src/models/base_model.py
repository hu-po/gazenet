import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils

'''
Base model class is inherited to re-use some common code
'''


class BaseModel(object):

    @base_utils.config_checker(['image_width',
                                'image_height',
                                'image_channels'])
    def __init__(self, config=None):
        # All models in this repo have the same image input
        self.image = tf.placeholder(tf.float32, shape=(None,
                                                       config.image_height,
                                                       config.image_width,
                                                       config.image_channels),
                                    name='input_image')
        # Boolean indicates whether model is in training mode
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        # List of summaries in this model class
        self.summaries = []

    def build_graph(self, config=None):
        self.predict = self.model_func(config=config)
        self.loss = self.loss_func(config=config)
        self.optimize = self.optimize_func(config=config)

    def model_func(self, config=None):
        raise NotImplementedError('Model must have a model function')

    def loss_func(self, config=None):
        raise NotImplementedError('Model must have a loss function')

    def add_summary(self, name, value, summary_type='scalar'):
        # Creates a summary object and adds it to the summaries list for the model class
        if summary_type is 'scalar':
            s = tf.summary.scalar(name, value)
        elif summary_type is 'image':
            s = tf.summary.image(name, value)
        else:
            raise Exception('Non valid summary type given')
        self.summaries.append(s)

    @base_utils.config_checker(['learning_rate', 'optimizer_type'])
    def optimize_func(self, config=None):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if config.optimizer_type == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
            elif config.optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            elif config.optimizer_type == 'adam':
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
            else:
                raise Exception('Unkown optimizer type: %s' % config.optimizer_type)
            # Add mean and variance ops to dependencies so batch norm works during training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(self.loss)
        return train_op
