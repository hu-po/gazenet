import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.models.base_model import BaseModel

'''
The discriminator network differentiates between synthetic and real images.
'''


class DiscriminatorModel(BaseModel):

    @base_utils.config_checker(['discrim_learning_rate',
                                'discrim_optimizer_type'])
    def __init__(self, config=None):
        # Reassign optimizer parameters
        config.learning_rate = config.discrim_learning_rate
        config.optimizer_type = config.discrim_optimizer_type
        super().__init__(config=config)
        self.label = tf.placeholder(tf.uint8, shape=(None, 2), name='label')
        # Placeholders for mixed batch
        self.real_image = tf.placeholder(tf.float32, shape=(None,
                                                            config.image_height,
                                                            config.image_width,
                                                            config.image_channels),
                                         name='real_images')
        self.refined_image = tf.placeholder(tf.float32, shape=(None,
                                                               config.image_height,
                                                               config.image_width,
                                                               config.image_channels),
                                            name='refined_images')

        with tf.variable_scope('discrim_model'):
            self.predict = self.model(config=config)
            self.loss = self.discrim_loss()
            self.optimize = self.optimizer(config=config)
            self.mixed_image_batch = self.create_mixed_image_batch()

    @base_utils.config_checker(['discrim_initializer'])
    def model(self, config=None):
        with tf.variable_scope('model', initializer=config.discrim_initializer,
                               reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            x = slim.conv2d(x, 96, [3, 3], stride=2, scope='conv1')
            x = slim.conv2d(x, 64, [3, 3], stride=2, scope='conv2')
            x = slim.max_pool2d(x, [3, 3], scope='pool1')
            x = slim.conv2d(x, 32, [3, 3], scope='conv3')
            x = slim.conv2d(x, 32, [1, 1], scope='conv4')
            x = slim.conv2d(x, 2, [1, 1], scope='conv5')
            x = slim.flatten(x)
            x = slim.fully_connected(x, 2, activation_fn=None)
            x = slim.softmax(x)
        return x

    def create_mixed_image_batch(self):
        # Shuffle together refined synthetic and real images in batch
        combined_images = tf.concat([self.real_image, self.refined_image], axis=0)
        # Create label vectors of same length as image batches (0=fake, 1=real)
        real_labels = tf.one_hot(tf.ones(shape=[tf.shape(self.real_image)[0]], dtype=tf.uint8), 2)
        fake_labels = tf.one_hot(tf.zeros(shape=[tf.shape(self.refined_image)[0]], dtype=tf.uint8), 2)
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)
        # Make sure to shuffle the images and labels with the same seed
        seed = 1
        shuffled_images = tf.random_shuffle(combined_images, seed=seed)
        shuffled_labels = tf.random_shuffle(combined_labels, seed=seed)
        return [shuffled_images, shuffled_labels]

    def discrim_loss(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.label,
                                                   logits=self.predict)
            self.add_summary('loss', loss)
        return loss
