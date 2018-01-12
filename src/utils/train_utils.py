import tensorflow as tf
from src.utils.base_utils import config_checker

'''
This file contains common functions used for training. Stored here to prevent
clutter in the main training files or in the general util file.
'''


@config_checker(['image_height', 'image_width', 'image_channels'])
def decode_image(serialized_example, config=None):
    """
    Decodes a serialized example for an image
    :param serialized_example: (parsed string Tensor) serialized example
    :param config: (Config) config object
    :return: image Tensor
    """
    features = tf.parse_single_example(
        serialized_example,
        features={'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.stack([config.image_height, config.image_width, config.image_channels_input])
    image = tf.reshape(image, image_shape)
    return image


@config_checker(['image_height', 'image_width', 'image_channels'])
def decode_gaze(serialized_example, config=None):
    """
    Decodes a serialized example for gaze images and labels
    :param serialized_example: (parsed string Tensor) serialized example
    :param config: (Config) config object
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
    image_shape = tf.stack([config.image_height, config.image_width, config.image_channels_input])
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
def grayscale(image, label=None, config=None):
    with tf.name_scope('image_prep'):
        if config.grayscale:
            image = tf.image.rgb_to_grayscale(image)
    if label is None:
        return image
    return image, label


@config_checker()
def standardize(image, label=None, config=None):
    with tf.name_scope('image_prep'):
        # Standardize the images
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        if label is None:
            return image
        # Standardize the labels
        label = tf.cast(label, tf.float32) * (1. / 100) - 0.5
    return image, label


@config_checker(['image_height',
                 'image_width',
                 'image_channels'])
def mixed_image_batch(self, config=None):
    # Placeholders for mixed batch
    real_images = tf.placeholder(tf.float32, shape=(None,
                                                    config.image_height,
                                                    config.image_width,
                                                    config.image_channels),
                                 name='real_images')
    refined_images = tf.placeholder(tf.float32, shape=(None,
                                                       config.image_height,
                                                       config.image_width,
                                                       config.image_channels),
                                    name='refined_images')
    # Combine together refined synthetic and real images in batch
    combined_images = tf.concat([real_images, refined_images], axis=0)
    # Create label vectors of same length as image batches (0=fake, 1=real)
    real_labels = tf.one_hot(tf.ones(shape=[tf.shape(self.real_image)[0]], dtype=tf.uint8), 2)
    fake_labels = tf.one_hot(tf.zeros(shape=[tf.shape(self.refined_image)[0]], dtype=tf.uint8), 2)
    combined_labels = tf.concat([real_labels, fake_labels], axis=0)
    # Make sure to shuffle the images and labels with the same seed
    seed = 1
    shuffled_images = tf.random_shuffle(combined_images, seed=seed)
    shuffled_labels = tf.random_shuffle(combined_labels, seed=seed)
    return real_images, refined_images, [shuffled_images, shuffled_labels]
