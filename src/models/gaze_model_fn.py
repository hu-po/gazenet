import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.models.layers as layers


def model_fn(features, labels, mode, params):
    """Model function for Estimator."""

    # Add training switch to parameters
    params['is_training'] = (mode == tf.estimator.ModeKeys.TRAIN)

    # Build model using layers defined in layers file
    x = layers.resnet(features['input_image'], params)
    x = layers.dim_reductor(x, params)
    x = layers.fc_head(x, params)

    # Final layer for regression has no activation function
    output = tf.layers.dense(x, 2, activation=None)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"gazeloc": output})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, output)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float64), output)
    }

    # Optimizer choices are made from parameter values
    train_op = layers.optimize(loss, params)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
