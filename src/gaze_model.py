import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
This network outputs the gaze location for a given webcam image.

The input:
* Webcam image (128x98x3)
* Target gaze location (2x1)

The output:
* A trained refiner network
'''


class GazeModel(object):

    def __init__(self, config):
        self.config = config
        if config.grayscale:
            self.config.image_channels = 1
        self.image = tf.placeholder(tf.float32, shape=(None,
                                                       config.image_height,
                                                       config.image_width,
                                                       config.image_channels),
                                    name='input_image')
        self.label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
        self.train_mode = tf.placeholder(tf.bool, shape=[], name='train_mode_switch')
        # Initialize properties
        self.predict
        self.mse
        self.optimize
        self.train_loss
        self.test_loss

    @property
    def predict(self):
        with tf.variable_scope('predict', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
            x = slim.conv2d(x, 32, [3, 3], scope='conv1')
            x = slim.conv2d(x, 64, [3, 3], scope='conv2')
            x = slim.conv2d(x, 64, [3, 3], scope='conv3')
            x = slim.max_pool2d(x, [2, 2], scope='pool1')
            x = slim.flatten(x)
            x = slim.dropout(x, self.config.dropout_keep_prob, is_training=self.train_mode)
            x = slim.fully_connected(x, 128)
            x = slim.dropout(x, self.config.dropout_keep_prob, is_training=self.train_mode)
            x = slim.fully_connected(x, 64)
            x = slim.dropout(x, self.config.dropout_keep_prob, is_training=self.train_mode)
            x = slim.fully_connected(x, 2, activation_fn=None)
        return x

    @property
    def optimize(self):
        with tf.variable_scope('optimize', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.mse)

    @property
    def mse(self):
        with tf.variable_scope('mse', reuse=tf.AUTO_REUSE):
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict)
        return mse

    @property
    def train_loss(self):
        tf.summary.scalar('train_loss', self.mse)
        return self.mse

    @property
    def test_loss(self):
        tf.summary.scalar('test_loss', self.mse)
        return self.mse
