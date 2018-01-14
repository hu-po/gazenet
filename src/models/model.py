import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.models.layers as layers

'''
Base model class is inherited to re-use some common code
'''


class Model(object):

    def __init__(self, config=None):
        self.graph = tf.Graph()
        self.config = config
        with self.graph.as_default():
            # All models in this repo have image input and labels
            self.image = tf.placeholder(tf.float32, shape=(None,
                                                           self.config.image_height,
                                                           self.config.image_width,
                                                           self.config.image_channels),
                                        name='input_image')
            self.label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
            # Boolean indicates whether model is in training mode
            self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
            # List of summaries in this model class
            self.summaries = []
            # Build graph
            self.predict = self.model_func()
            self.loss = self.loss_func()
            self.optimize = self.optimize_func()

    # TODO: Load saved modelsl head function')
    def model_base(self, x):
        x = layers.resnet(x, self)
        return x

    def model_head(self, input):
        raise NotImplementedError('Model must have a model head function')

    def model_func(self):
        with tf.variable_scope(self.config.model_name, initializer=self.config.initializer, reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            x = self.model_base(x)
            x = self.model_head(x)
            return x

    def loss_func(self):
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

    def optimize_func(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.config.optimizer_type == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
            elif self.config.optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
            elif self.config.optimizer_type == 'adam':
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            else:
                raise Exception('Unkown optimizer type: %s' % self.config.optimizer_type)
            # Add mean and variance ops to dependencies so batch norm works during training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.config.model_name)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(self.loss)
        return train_op
