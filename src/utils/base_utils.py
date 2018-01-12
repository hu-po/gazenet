import re
import os
import glob
from PIL import Image
import tensorflow as tf
import numpy as np

'''
This file file contains common functions used for dataset manipulation, config checking, etc
'''


def config_checker(config_properties=None):
    """
    Decorator checks to make sure the function contains the neccessary config values
    :param config_properties: [string] list of strings of properties used in function
    :return: function
    """

    def decorator(func):
        def wrapped(*args, **kwargs):
            assert kwargs.get('config', None) is not None, '%s needs config argument' % func.__name__
            for prop in (config_properties or []):
                assert kwargs['config'].__getattribute__(prop) is not None, \
                    '%s needs the (not None) property %s' % (func.__name__, prop)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


@config_checker(['filename_regex'])
def _extract_target_from_gazefilename(path_string, config=None):
    # Extract the label from the image path name
    m = re.search(config.filename_regex, os.path.basename(path_string))
    gaze_x = int(float(m.group(1)) * 100)
    gaze_y = int(float(m.group(2)) * 100)
    return gaze_x, gaze_y


@config_checker(['image_width', 'image_height'])
def _imagepath_to_string(image_path, config=None):
    # Get image and label from image path
    image_raw = Image.open(image_path)
    image_resized = image_raw.resize((config.image_width, config.image_height))
    img = np.array(image_resized)
    # Sometimes images have an extra 4th alpha channel
    img = img[:, :, :3]
    return img.tostring()


@config_checker()
def _write_gazedata_tfrecord(tfrecord_path, image_paths, config=None):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for image_path in image_paths:
        img_string = _imagepath_to_string(image_path, config=config)
        gaze_x, gaze_y = _extract_target_from_gazefilename(image_path, config=config)
        # Feature defines each discrete entry in the tfrecords file
        feature = {
            'gaze_x': _int64_feature(gaze_x),
            'gaze_y': _int64_feature(gaze_y),
            'image_raw': _bytes_feature(img_string),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


@config_checker(['dataset_path', 'train_test_split', 'train_tfrecord_path', 'test_tfrecord_path'])
def gazedata_to_tfrecords(config=None):
    if os.path.exists(config.train_tfrecord_path) or os.path.exists(config.test_tfrecord_path):
        print('TFRecords have already been created for this dataset')
        return
    image_paths = glob.glob(os.path.join(config.dataset_path, '*.png'))
    # Split image paths into test and train
    total_images = len(image_paths)
    num_train = int(config.train_test_split * total_images)
    num_test = total_images - num_train
    print('There are %d images in %s. Using a %0.2f split, we get %d train and %d test' %
          (total_images, config.dataset_name, config.train_test_split, num_train, num_test))
    # Add each image to either test or train list
    test_idx = np.random.choice(total_images, num_test)
    train_image_paths = []
    test_image_paths = []
    for i, path in enumerate(image_paths):
        if i in test_idx:
            train_image_paths.append(path)
        else:
            test_image_paths.append(path)
    # Write train and test tfrecords to paths in config
    _write_gazedata_tfrecord(config.train_tfrecord_path, train_image_paths, config=config)
    _write_gazedata_tfrecord(config.test_tfrecord_path, test_image_paths, config=config)


@config_checker()
def _write_image_tfrecord(tfrecord_path, image_paths, config=None):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for image_path in image_paths:
        img_string = _imagepath_to_string(image_path, config=config)
        # Feature defines each discrete entry in the tfrecords file
        feature = {
            'image_raw': _bytes_feature(img_string),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


@config_checker(['tfrecord_path',
                 'dataset_path',
                 'dataset_name'])
def image_to_tfrecords(config=None):
    if os.path.exists(config.tfrecord_path):
        print('TFRecords has already been created for this dataset')
        return
    image_paths = glob.glob(os.path.join(config.dataset_path, '*.png'))
    print('There are %d images in %s' % (len(image_paths), config.dataset_name))

    # Write synthetic and real tfrecords to paths in config
    _write_image_tfrecord(config.tfrecord_path, image_paths, config=config)
