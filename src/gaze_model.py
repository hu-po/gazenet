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
                                                       config.image_width,
                                                       config.image_height,
                                                       config.image_channels))
        self.label = tf.placeholder(tf.float32, shape=(None, 2))
        self.predict = self.predict_func()
        self.mse = self.mse_func()
        self.optimize = self.optimize_func()

    def predict_func(self):
        with tf.variable_scope('predict', initializer=slim.xavier_initializer(), reuse=tf.AUTO_REUSE):
            x = self.image
            tf.summary.image('input_image', x)
            x = slim.conv2d(x, 32, [3, 3], scope='conv1')
            x = slim.conv2d(x, 64, [3, 3], scope='conv2')
            x = slim.conv2d(x, 64, [3, 3], scope='conv3')
            x = slim.max_pool2d(x, [2, 2], scope='pool1')
            x = slim.flatten(x)
            x = slim.fully_connected(x, 128)
            x = slim.fully_connected(x, 64)
            x = slim.fully_connected(x, 2)
        return x

    def optimize_func(self):
        with tf.variable_scope('optimize', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.mse)

    def mse_func(self):
        with tf.variable_scope('mse', reuse=tf.AUTO_REUSE):
            mse = tf.losses.mean_squared_error(labels=self.label, predictions=self.predict)
            tf.summary.scalar('mse', mse)
        return mse
