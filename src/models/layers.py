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


def optimize(loss, params):
    if params['optimizer_type'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(params["learning_rate"])
    elif params['optimizer_type'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
    elif params['optimizer_type'] == 'adam':
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    else:
        raise Exception('Unkown optimizer type: %s' % params['optimizer_type'])
    # Add mean and variance ops to dependencies so batch norm works during training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=params["model_name"])
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op
