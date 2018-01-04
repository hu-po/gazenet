import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys

mod_path = os.path.abspath(os.path.join('..'))
sys.path.append(mod_path)

from src.helper_func import define_scope

'''

This network outputs the gaze location for a given webcam image.

The input:
* Webcam image (128x128x3)
* Target gaze location (quadrant?)

The output:
* A trained refiner network

'''

class GazeModel(object):

    def __init__(self, image, label, config): #test_image, test_label, config):
        self.image = image
        self.label = label
        # self.test_image = test_image
        # self.test_label = test_label
        self.config = config
        self.predict
        self.optimize
        self.loss

    @define_scope(initializer=slim.xavier_initializer())
    def predict(self):
        x = self.image
        x = slim.conv2d(x, 32, [3, 3], scope='conv1')
        x = slim.conv2d(x, 64, [3, 3], scope='conv2')
        x = slim.max_pool2d(x, [2, 2], scope='pool1')
        x = slim.flatten(x)
        x = slim.fully_connected(x, 128)
        x = slim.fully_connected(x, 64)
        x = slim.fully_connected(x, self.config['output_classes'], tf.nn.softmax)
        return x

    @define_scope
    def optimize(self):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label,
                                                      logits=self.predict)
        optimizer = tf.train.RMSPropOptimizer(self.config['learning_rate'])
        return optimizer.minimize(loss)

    @define_scope
    def loss(self):
        predicted_label = tf.cast(tf.argmax(self.predict, 1), tf.int32)
        mistakes = tf.not_equal(self.label, predicted_label)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
