import tensorflow as tf
import tensorflow.contrib.slim as slim

'''

This network outputs the gaze location for a given webcam image.

The input:
* Webcam image (128x128x3)
* Target gaze location (quadrant?)

The output:
* A trained refiner network

'''


class GazeModel(object):

    def __init__(self, image, label, config):
        self.image = image
        self.label = label
        self.config = config
        self.predict = self.predict_func()
        self.optimize = self.optimize_func()
        self.accuracy = self.accuracy_func()

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
            x = slim.fully_connected(x, self.config['output_classes'], tf.nn.softmax)
        return x

    def optimize_func(self):
        with tf.variable_scope('optimize', reuse=tf.AUTO_REUSE):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label, logits=self.predict)
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.RMSPropOptimizer(self.config['learning_rate'])
        return optimizer.minimize(loss)

    def accuracy_func(self):
        with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
            predicted_label = tf.cast(tf.argmax(self.predict, 1), tf.int32)
            mistakes = tf.not_equal(self.label, predicted_label)
            accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return accuracy
