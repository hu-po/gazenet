import os
import sys
import glob
import re
import tensorflow as tf
import PIL.Image
import numpy as np

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import Config
import src.utils.train_utils as train_utils


class Dataset(object):

    @classmethod
    def from_yaml(cls, config):
        assert config is not None, 'Please provide a yaml config file for the dataset'
        cls.config = Config.from_yaml(config)
        assert cls.config.dataset_type in ['gaze', 'image'], 'Config dataset type unknown or missing'
        if cls.config.dataset_type == 'gaze':
            return GazeDataset()
        if cls.config.dataset_type == 'image':
            return ImageDataset()
        return cls

    def __init__(self):
        self.dataset_path = os.path.join(self.config.data_dir, self.config.dataset_name)
        if not os.path.exists(self.dataset_path):
            raise Exception('Dataset not found in path')
        self.tfrecord_path = os.path.join(self.dataset_path, self.config.tfrecord_name)
        if os.path.exists(self.tfrecord_path):
            print('TFRecords has already been created for %s' % self.config.dataset_name)
        else:
            self._to_tfrecords()

    # Define the input feed function in here, TF is picky about what it wants
    def input_feed(self):
        with tf.name_scope('input_feed_gen'):
            dataset = tf.data.TFRecordDataset(self.tfrecord_path)
            dataset = dataset.take(self.config.dataset_len)
            dataset = dataset.map(lambda x: self._decode(x))
            # dataset.repeat()  # Repeat dataset indefinitely
            if self.config.image_augmentation:
                dataset = dataset.map(lambda *x: self._image_augmentation(x))
            if self.config.grayscale:
                dataset = dataset.map(lambda *x: self._grayscale(x))
            dataset = dataset.map(lambda *x: self._standardize(x))
            if self.config.shuffle:
                dataset = dataset.shuffle(self.config.buffer_size)
            dataset = dataset.batch(self.config.batch_size)
            iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    def _to_tfrecords(self):
        image_paths = glob.glob(os.path.join(self.dataset_path, '*.png'))
        print('There are %d images in %s' % (len(image_paths), self.config.dataset_name))
        writer = tf.python_io.TFRecordWriter(self.tfrecord_path)
        self._write_tfrecords(image_paths, writer)
        writer.close()

    def _imagepath_to_string(self, image_path):
        # Get image and label from image path
        image_raw = PIL.Image.open(image_path)
        image_resized = image_raw.resize((self.config.image_width, self.config.image_height))
        img = np.array(image_resized)
        # Sometimes images have an extra 4th alpha channel
        img = img[:, :, :3]
        return img.tostring()

    def _image_augmentation(self, image):
        with tf.name_scope('image_augment'):
            if self.config.random_brigtness:
                image = tf.image.random_brightness(image, self.config.brightnes_max_delta)
            if self.config.random_contrast:
                image = tf.image.random_contrast(image, self.config.contrast_lower, self.config.contrast_upper)
            return image

    def _grayscale(self, image):
        with tf.name_scope('grayscale'):
            image = tf.image.rgb_to_grayscale(image)
            return image

    def _standardize(self, image):
        with tf.name_scope('image_prep'):
            image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
            return image

    def _write_tfrecords(self, image_paths, writer):
        raise NotImplementedError('Implemented in subclasses')

    def _decode(self, serialized_example):
        raise NotImplementedError('Implemented in subclasses')

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ImageDataset(Dataset):

    def _decode(self, serialized_example):
        """
        Decodes a serialized example for an image
        :param serialized_example: (parsed string Tensor) serialized example
        :return: image Tensor
        """
        features = tf.parse_single_example(
            serialized_example,
            features={'image_raw': tf.FixedLenFeature([], tf.string)})
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image_shape = tf.stack([self.config.image_height, self.config.image_width, self.config.image_channels_input])
        image = tf.reshape(image, image_shape)
        return image

    def _write_tfrecords(self, image_paths, writer):
        for image_path in image_paths:
            img_string = self._imagepath_to_string(image_path)
            # Feature defines each discrete entry in the tfrecords file
            feature = {
                'image_raw': self._bytes_feature(img_string),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


class GazeDataset(Dataset):

    def _decode(self, serialized_example):
        """
        Decodes a serialized example for gaze images and labels
        :param serialized_example: (parsed string Tensor) serialized example
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
        image_shape = tf.stack([self.config.image_height, self.config.image_width, self.config.image_channels_input])
        image = tf.reshape(image, image_shape)
        return image, target

    def _write_tfrecords(self, image_paths, writer):
        for image_path in image_paths:
            img_string = self._imagepath_to_string(image_path)
            gaze_x, gaze_y = self._extract_target_from_gazefilename(image_path)
            # Feature defines each discrete entry in the tfrecords file
            feature = {
                'gaze_x': self._int64_feature(gaze_x),
                'gaze_y': self._int64_feature(gaze_y),
                'image_raw': self._bytes_feature(img_string),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    def _extract_target_from_gazefilename(self, path_string):
        # Extract the label from the image path name
        m = re.search(self.config.filename_regex, os.path.basename(path_string))
        gaze_x = int(float(m.group(1)) * 100)
        gaze_y = int(float(m.group(2)) * 100)
        return gaze_x, gaze_y

    def _image_augmentation(self, *args):
        image = super()._image_augmentation(args[0][0])
        return image, args[0][1]

    def _grayscale(self, *args):
        image = super()._grayscale(args[0][0])
        return image, args[0][1]

    def _standardize(self, *args):
        image = super()._standardize(args[0][0])
        label = tf.cast(args[0][1], tf.float32) * (1. / 100) - 0.5
        return image, label
