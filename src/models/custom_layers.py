import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils


@base_utils.config_checker(['fc_layers',
                            'dropout_keep_prob'])
def fc_head(input, model, config=None):
    x = input
    for i in range(len(config.fc_layers)):
        x = slim.fully_connected(x, config.fc_layers[i])
        x = slim.dropout(x, config.dropout_keep_prob, is_training=model.is_training)
    return x


@base_utils.config_checker(['dropout_keep_prob',
                            'batch_norm',
                            'dimred_feat',
                            'dimred_kernel',
                            'dimred_stride'])
def dim_reductor(input, model, config=None):
    x = slim.conv2d(input, config.dimred_feat, config.dimred_kernel, stride=config.dimred_stride)
    if config.batch_norm:
        x = tf.layers.batch_normalization(x, training=model.is_training)
    x = slim.flatten(x)
    x = slim.dropout(x, config.dropout_keep_prob, is_training=model.is_training)
    return x


@base_utils.config_checker(['rb_feat',
                            'rb_kernel',
                            'batch_norm'])
def resnet_block(input, model, config=None):
    x = slim.conv2d(input, config.rb_feat, config.rb_kernel, padding='same')
    if config.batch_norm:
        x = tf.layers.batch_normalization(x, training=model.is_training)
    x = tf.concat([x, input], axis=3)
    return x


@base_utils.config_checker(['num_rb'])
def resnet(input, model, config=None):
    x = input
    for _ in range(config.num_rb):
        x = resnet_block(x, model, config=config)
    return x
