import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim


def fc_head(input, model):
    x = input
    for i in range(len(model.config.fc_layers)):
        x = slim.fully_connected(x, model.config.fc_layers[i])
        x = slim.dropout(x, model.config.dropout_keep_prob, is_training=model.is_training)
    return x


def dim_reductor(input, model):
    x = slim.conv2d(input, model.config.dimred_feat, model.config.dimred_kernel, stride=model.config.dimred_stride)
    if model.config.batch_norm:
        x = tf.layers.batch_normalization(x, training=model.is_training)
    x = slim.flatten(x)
    x = slim.dropout(x, model.config.dropout_keep_prob, is_training=model.is_training)
    return x


def resnet_block(input, model):
    x = slim.conv2d(input, model.config.rb_feat, model.config.rb_kernel, padding='same')
    if model.config.batch_norm:
        x = tf.layers.batch_normalization(x, training=model.is_training)
    x = tf.concat([x, input], axis=3)
    return x


def resnet(input, model):
    with tf.variable_scope('resnet'):
        x = input
        for _ in range(model.config.num_rb):
            x = resnet_block(x, model)
    return x
