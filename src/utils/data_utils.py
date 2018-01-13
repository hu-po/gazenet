import re
import glob
import os
import sys
import tensorflow as tf
from PIL import Image
import numpy as np

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import config_checker

'''
This file file contains common functions used for dataset manipulation
'''


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
def _write_gazedata_tfrecord(image_paths, writer, config=None):
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


@config_checker()
def _write_image_tfrecord(image_paths, writer, config=None):
    for image_path in image_paths:
        img_string = _imagepath_to_string(image_path, config=config)
        # Feature defines each discrete entry in the tfrecords file
        feature = {
            'image_raw': _bytes_feature(img_string),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


@config_checker(['dataset_path', 'dataset_name', 'dataset_type'])
def to_tfrecords(config=None):
    if os.path.exists(config.tfrecord_path):
        print('TFRecords has already been created for this dataset')
        return
    image_paths = glob.glob(os.path.join(config.dataset_path, '*.png'))
    print('There are %d images in %s' % (len(image_paths), config.dataset_name))
    writer = tf.python_io.TFRecordWriter(config.tfrecord_path)
    if config.dataset_type == 'image':
        _write_image_tfrecord(image_paths, writer, config=config)
    elif config.dataset_type == 'gaze':
        _write_gazedata_tfrecord(image_paths, writer, config=config)
    else:
        raise Exception('Need to provide utils for this datatype')
    writer.close()
