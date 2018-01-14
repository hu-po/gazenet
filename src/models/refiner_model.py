import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.models.model import Model

'''
The refiner network 'refines' a synthetic image, making it more real.
'''


class RefinerModel(Model):

    def model_head(self, x):
        x = slim.conv2d(x, self.config.image_channels, [1, 1], scope='final_refiner_conv')
        self.add_summary('output_image', x, 'image')
        return x

    def loss_func(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # All images are fake (synthetic), so labels are all 0
            labels = tf.zeros(shape=[tf.shape(self.label)[0]], dtype=tf.uint8)
            one_hot_labels = tf.one_hot(labels, 2)
            loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=one_hot_labels,
                                                        logits=self.label,
                                                        scope='loss_real')
            loss_reg = tf.losses.absolute_difference(predictions=self.image,
                                                     labels=self.predict,
                                                     weights=self.config.regularization_lambda,
                                                     scope='loss_reg')
            loss = tf.add(loss_real, loss_reg)
            self.add_summary('loss_real', loss_real, 'scalar')
            self.add_summary('loss_reg', loss_reg, 'scalar')
            self.add_summary('loss', loss, 'scalar')
        return loss
