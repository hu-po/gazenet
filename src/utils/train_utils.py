import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import config_checker

'''
This file contains common functions used for training (input feeding, data decoding and augmentation, etc)
'''


@config_checker(['image_height', 'image_width', 'image_channels'])
def _decode_image(serialized_example, config=None):
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
def _decode_gaze(serialized_example, config=None):
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


@config_checker(['dataset_type', 'random_brigtness', 'random_contrast'])
def _image_augmentation(*args, config=None):
    with tf.name_scope('image_augment'):
        image = args[0]
        if config.random_brigtness:
            image = tf.image.random_brightness(image, config.brightnes_max_delta)
        if config.random_contrast:
            image = tf.image.random_contrast(image, config.contrast_lower, config.contrast_upper)
        if config.dataset_type == 'gaze':
            return image, args[1]
        return image


@config_checker(['dataset_type', 'grayscale'])
def _grayscale(*args, config=None):
    with tf.name_scope('grayscale'):
        image = args[0]
        image = tf.image.rgb_to_grayscale(image)
        if config.dataset_type == 'gaze':
            return image, args[1]
        return image


@config_checker(['dataset_type'])
def _standardize(*args, config=None):
    with tf.name_scope('image_prep'):
        image = args[0]
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        if config.dataset_type == 'gaze':
            label = args[1]
            label = tf.cast(label, tf.float32) * (1. / 100) - 0.5
            return image, label
        return image


@config_checker(['dataset_type', 'dataset_len', 'tfrecord_path', 'image_augmentation', 'shuffle', 'batch_size'])
def input_feed(config=None):
    with tf.name_scope('input_feed_gen'):
        dataset = tf.data.TFRecordDataset(config.tfrecord_path)
        dataset = dataset.take(config.dataset_len)
        if config.dataset_type == 'image':
            dataset = dataset.map(lambda x: _decode_image(x, config=config))
            dataset = dataset.repeat()  # Repeat dataset indefinitely
        elif config.dataset_type == 'gaze':
            dataset = dataset.map(lambda x: _decode_gaze(x, config=config))
        else:
            raise Exception('Need to provide train utils for this dataset type')
        if config.image_augmentation:
            dataset = dataset.map(lambda x: _image_augmentation(x, config=config))
        if config.grayscale:
            dataset = dataset.map(lambda x: _grayscale(x, config=config))
        dataset = dataset.map(lambda x: _standardize(x, config=config))
        if config.shuffle:
            dataset = dataset.shuffle(config.buffer_size)
        dataset = dataset.batch(config.batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@config_checker(['image_height', 'image_width', 'image_channels'])
def mixed_image_batch(config=None):
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
    real_labels = tf.one_hot(tf.ones(shape=[tf.shape(real_images)[0]], dtype=tf.uint8), 2)
    fake_labels = tf.one_hot(tf.zeros(shape=[tf.shape(refined_images)[0]], dtype=tf.uint8), 2)
    combined_labels = tf.concat([real_labels, fake_labels], axis=0)
    # Make sure to shuffle the images and labels with the same seed
    seed = 1
    shuffled_images = tf.random_shuffle(combined_images, seed=seed)
    shuffled_labels = tf.random_shuffle(combined_labels, seed=seed)
    return real_images, refined_images, [shuffled_images, shuffled_labels]
