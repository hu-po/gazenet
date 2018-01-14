import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import config_checker
from src.models.model import BaseModel
import src.models.layers as layers

'''
The discriminator network differentiates between synthetic and real images.
'''


class DiscriminatorModel(BaseModel):

    @config_checker()
    def __init__(self, config=None):
        super().__init__(config=config)
        with self.graph.as_default():
            self.label = tf.placeholder(tf.uint8, shape=(None, 2), name='label')
            self.build_graph()

    def model_base(self, x):
        x = layers.resnet(x, self)
        return x

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
