import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils
from src.models.base_model import BaseModel
import src.models.custom_layers as layers

'''
The discriminator network differentiates between synthetic and real images.
'''


class DiscriminatorModel(BaseModel):

    @base_utils.config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        self.label = tf.placeholder(tf.uint8, shape=(None, 2), name='label')
        self.build_graph(config=config)

    @base_utils.config_checker(['initializer'])
    def model_func(self, config=None):
        with tf.variable_scope('discrim_model', initializer=config.initializer, reuse=tf.AUTO_REUSE):
            x = self.image
            self.add_summary('input_image', x, 'image')
            # Model arch is a stack of conv blocks with residual connections, then a fully connected head
            x = layers.resnet(x, self, config=config)
            x = layers.dim_reductor(x, self, config=config)
            x = layers.fc_head(x, self, config=config)
            x = slim.fully_connected(x, 2, activation_fn=None)
            x = slim.softmax(x, scope='output')
        return x

    @base_utils.config_checker()
    def loss_func(self, config=None):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.label, logits=self.predict)
            self.add_summary('loss', loss, 'scalar')
        return loss
