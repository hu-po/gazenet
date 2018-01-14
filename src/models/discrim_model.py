import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.models.model import Model
import src.models.layers as layers

'''
The discriminator network differentiates between synthetic and real images.
'''


class DiscriminatorModel(Model):

    def model_head(self, x):
        x = layers.dim_reductor(x, self)
        x = layers.fc_head(x, self)
        x = slim.fully_connected(x, 2, activation_fn=None)
        x = slim.softmax(x, scope='output')
        return x

    def loss_func(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.label, logits=self.predict)
            self.add_summary('loss', loss, 'scalar')
        return loss
