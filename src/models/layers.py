import tensorflow as tf


def fc_head(input, params):
    x = input
    for i in range(len(params['fc_layers'])):
        x = slim.fully_connected(x, model.config.fc_layers[i])
        x = slim.dropout(x, model.config.dropout_keep_prob, is_training=model.is_training)
    return x


def dim_reductor(input, params):
    x = tf.layers.conv2d(input, model.config.dimred_feat, model.config.dimred_kernel, stride=model.config.dimred_stride)
    if model.config.batch_norm:
        x = tf.layers.batch_normalization(x, training=model.is_training)
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, model.config.dropout_keep_prob, training=model.is_training)
    return x


def resnet_block(input, params):
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