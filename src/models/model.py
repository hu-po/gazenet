import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.config.config import config_checker

'''
Base model class is inherited to re-use some common code
'''


class BaseModel(object):

    @config_checker(['image_width',
                     'image_height',
                     'image_channels'])
    def __init__(self, config=None):
        self.graph = tf.Graph()
        self.config = config
        with self.graph.as_default():
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

    def build_graph(self):
        self.predict = self.model_func()
        self.loss = self.loss_func()
        self.optimize = self.optimize_func()

    def model_base(self, input):
        raise NotImplementedError('Model must have a model base function')

    def model_head(self, input):
        raise NotImplementedError('Model must have a model head function')

    def model_func(self):
        with tf.variable_scope(self.config.model_name, initializer=self.config.initializer, reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            x = self.base_func(x)
            x = self.head_func(x)
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
