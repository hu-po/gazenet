import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.models.layers as layers


def optimize(loss, params):
    if params['optimizer_type'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(params['learning_rate'])
    elif params['optimizer_type'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(params['learning_rate'])
    elif params['optimizer_type'] == 'adam':
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    else:
        raise Exception('Unkown optimizer type: %s' % params['optimizer_type'])
    # Add mean and variance ops to dependencies so batch norm works during training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=params['model_name'])
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op


def resnet_gaze_model_fn(features, labels, mode, params):

    # Add training switch to parameters
    params['is_training'] = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope(params['model_name']):
        # Build model using layers defined in layers file
        input_image = features
        x = layers.resnet(input_image, params)
        x = layers.dim_reductor(x, params)
        x = layers.fc_head(x, params)

        # Final layer for regression has no activation function
        output = tf.layers.dense(x, 2, activation=None, name='output')

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'gazeloc': output})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, output)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels, output)
    }

    # Optimizer choices are made from parameter values
    train_op = optimize(loss, params)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
