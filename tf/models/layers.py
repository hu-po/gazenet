import tensorflow as tf


def fc_head(input, params):
    x = input
    for i in range(len(params['fc_layers'])):
        x = tf.layers.dense(x, params['fc_layers'][i])
        x = tf.layers.dropout(x, params['dropout_keep_prob'], training=params['is_training'])
    return x


def dim_reductor(input, params):
    x = tf.layers.conv2d(input, params['dimred_feat'], params['dimred_kernel'], strides=params['dimred_stride'])
    if params['batch_norm']:
        x = tf.layers.batch_normalization(x, training=params['is_training'])
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, params['dropout_keep_prob'], training=params['is_training'])
    return x


def resnet_block(input, params):
    x = tf.layers.conv2d(input, params['rb_feat'], params['rb_kernel'], padding='same')
    if params['batch_norm']:
        x = tf.layers.batch_normalization(x, training=params['is_training'])
    x = tf.concat([x, input], axis=3)
    return x


def resnet(input, params):
    with tf.variable_scope('resnet'):
        x = input
        for _ in range(params['num_rb']):
            x = resnet_block(x, params)
    return x
