import os
import sys
import glob
import re
import tensorflow as tf
import PIL as Image
import numpy as np
import yaml

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.grandpa import Grandpa


class Dataset(Grandpa):

    def __init__(self, config=None):
        assert config is not None, 'Please provide a yaml config file for the dataset'
        self.config = yaml.load(config)
        self.dataset_path = os.path.join(self.data_dir, self.cnf.dataset_name)
        if not os.path.exists(self.dataset_path):
            raise Exception('Dataset not found in path')
        self.tfrecord_path = os.path.join(self.dataset_path, self.tfrecord_name)
        if os.path.exists(self.tfrecord_path):
            print('TFRecords has already been created for %s' % self.dataset_name)
        else:
            self._to_tfrecords()

    def input_feed(self):
        with tf.name_scope('input_feed_gen'):
            dataset = tf.data.TFRecordDataset(self.tfrecord_path)
            dataset = dataset.take(self.dataset_len)
            if self.dataset_type == 'image':
                dataset = dataset.map(lambda x: self._decode_image(x))
                dataset = dataset.repeat()  # Repeat dataset indefinitely
            elif self.dataset_type == 'gaze':
                dataset = dataset.map(lambda x: self._decode_gaze(x))
            else:
                raise Exception('Need to provide train utils for this dataset type')
            if self.image_augmentation:
                dataset = dataset.map(lambda *x: self._image_augmentation(x))
            if self.grayscale:
                dataset = dataset.map(lambda *x: self._grayscale(x))
            dataset = dataset.map(lambda *x: self._standardize(x))
            if self.shuffle:
                dataset = dataset.shuffle(self.buffer_size)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
        return iterator, iterator.get_next()

    def mixed_image_batch(self):
        # Placeholders for mixed batch
        real_images = tf.placeholder(tf.float32, shape=(None,
                                                        self.image_height,
                                                        self.image_width,
                                                        self.image_channels),
                                     name='real_images')
        refined_images = tf.placeholder(tf.float32, shape=(None,
                                                           self.image_height,
                                                           self.image_width,
                                                           self.image_channels),
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

    def _to_tfrecords(self):
        image_paths = glob.glob(os.path.join(self.dataset_path, '*.png'))
        print('There are %d images in %s' % (len(image_paths), self.dataset_name))
        writer = tf.python_io.TFRecordWriter(self.tfrecord_path)
        if self.dataset_type == 'image':
            self._write_image_tfrecord(image_paths, writer)
        elif self.dataset_type == 'gaze':
            self._write_gazedata_tfrecord(image_paths, writer)
        else:
            raise Exception('Need to provide utils for this datatype')
        writer.close()

    def _extract_target_from_gazefilename(self, path_string):
        # Extract the label from the image path name
        m = re.search(self.filename_regex, os.path.basename(path_string))
        gaze_x = int(float(m.group(1)) * 100)
        gaze_y = int(float(m.group(2)) * 100)
        return gaze_x, gaze_y

    def _imagepath_to_string(self, image_path):
        # Get image and label from image path
        image_raw = Image.open(image_path)
        image_resized = image_raw.resize((self.image_width, self.image_height))
        img = np.array(image_resized)
        # Sometimes images have an extra 4th alpha channel
        img = img[:, :, :3]
        return img.tostring()

    def _write_gazedata_tfrecord(self, image_paths, writer):
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

    def _write_image_tfrecord(self, image_paths, writer):
        for image_path in image_paths:
            img_string = self._imagepath_to_string(image_path)
            # Feature defines each discrete entry in the tfrecords file
            feature = {
                'image_raw': self._bytes_feature(img_string),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    def _decode_image(self, serialized_example):
        """
        Decodes a serialized example for an image
        :param serialized_example: (parsed string Tensor) serialized example
        :return: image Tensor
        """
        features = tf.parse_single_example(
            serialized_example,
            features={'image_raw': tf.FixedLenFeature([], tf.string)})
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image_shape = tf.stack([self.image_height, self.image_width, self.image_channels])
        image = tf.reshape(image, image_shape)
        return image

    def _decode_gaze(self, serialized_example):
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
        image_shape = tf.stack([self.image_height, self.image_width, self.image_channels])
        image = tf.reshape(image, image_shape)
        return image, target

    def _image_augmentation(self, *args):
        with tf.name_scope('image_augment'):
            image = args[0][0]
            if self.random_brigtness:
                image = tf.image.random_brightness(image, self.brightnes_max_delta)
            if self.random_contrast:
                image = tf.image.random_contrast(image, self.contrast_lower, self.contrast_upper)
            if self.dataset_type == 'gaze':
                return image, args[0][1]
            return image

    def _grayscale(self, *args):
        with tf.name_scope('grayscale'):
            image = args[0][0]
            image = tf.image.rgb_to_grayscale(image)
            if self.dataset_type == 'gaze':
                return image, args[0][1]
            return image

    def _standardize(self, *args):
        with tf.name_scope('image_prep'):
            image = args[0][0]
            image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
            if self.dataset_type == 'gaze':
                label = args[0][1]
                label = tf.cast(label, tf.float32) * (1. / 100) - 0.5
                return image, label
            return image

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
