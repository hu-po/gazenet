import os
import sys
import time
import datetime
import tensorflow as tf

'''
This file contains common functions used for training. Stored here to prevent
clutter in the main training files.
'''


def config_checker(config_properties):
    """
    Decorator checks to make sure the function contains the neccessary config values
    :param config_properties: [string] list of strings of properties used in function
    :return: function
    """
    def decorator(func):
        def wrapped(*args, **kwargs):
            assert kwargs.get('config', None) is not None, '%s needs config argument' % func.__name__
            for prop in config_properties:
                assert kwargs['config'].__getattribute__(prop) is not None,\
                    '%s needs the (not None) property %s' % (func.__name__, prop)
            return func(*args, **kwargs)
        return wrapped
    return decorator


@config_checker(['image_height', 'image_width', 'image_channels'])
def decode_gaze(serialized_example, config=None):
    """
    Decodes a serialized example for gaze images and labels
    :param serialized_example: (parsed string Tensor) serialized example
    :param config: config namespace
    :return: image and target Tensors
    """
    features = tf.parse_single_example(
        serialized_example,
        features={
            'gaze_x': tf.FixedLenFeature([], tf.int64),
            'gaze_y': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    gaze_x = tf.cast(features['gaze_x'], tf.int32)
    gaze_y = tf.cast(features['gaze_y'], tf.int32)
    target = [gaze_x, gaze_y]
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.stack([config.image_height, config.image_width, config.image_channels])
    image = tf.reshape(image, image_shape)
    return image, target


@config_checker(['random_brigtness', 'brightnes_max_delta',
                 'random_contrast', 'contrast_lower', 'contrast_upper'])
def image_augmentation(image, label, config=None):
    with tf.name_scope('image_augment'):
        # Apply image adjustments to reduce overfitting
        if config.random_brigtness:
            image = tf.image.random_brightness(image, config.brightnes_max_delta)
        if config.random_contrast:
            image = tf.image.random_contrast(image, config.contrast_lower, config.contrast_upper)
    return image, label


@config_checker(['grayscale'])
def grayscale(image, label, config=None):
    with tf.name_scope('image_prep'):
        if config.grayscale:
            image = tf.image.rgb_to_grayscale(image)
    return image, label


def standardize(image, label, config=None):
    with tf.name_scope('image_prep'):
        # Standardize the images
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        # Standardize the labels
        label = tf.cast(label, tf.float32) * (1. / 100) - 0.5
    return image, label
