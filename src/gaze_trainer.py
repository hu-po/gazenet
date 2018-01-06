import os
import sys
import argparse
import time
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.gaze_model import GazeModel
import src.config as config

'''
This file is used to train the gaze net. It contains functions for reading
and decoding the data (should be in TFRecords format).
'''


def _decode(serialized_example):
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
    image_shape = tf.stack([config.image_width, config.image_height, config.image_channels])
    image = tf.reshape(image, image_shape)
    return image, target


def _image_augmentation(image, label):
    with tf.name_scope('image_augment'):
        # Apply image adjustments to reduce overfitting
        if config.random_brigtness:
            image = tf.image.random_brightness(image, config.brightnes_max_delta)
        if config.random_contrast:
            image = tf.image.random_contrast(image, config.contrast_lower, config.contrast_upper)
    return image, label


def _image_prep(image, label):
    with tf.name_scope('image_prep'):
        if config.grayscale:
            image = tf.image.rgb_to_grayscale(image)
        # Standardize the images
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


def _train_iterator():
    with tf.name_scope('train_input'):
        dataset = tf.data.TFRecordDataset(config.train_tfrecord_path)
        dataset = dataset.map(_decode)
        dataset = dataset.map(_image_augmentation)
        dataset = dataset.map(_image_prep)
        dataset = dataset.shuffle(config.buffer_size)
        dataset = dataset.batch(config.batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator, dataset


def _test_iterator():
    with tf.name_scope('test_input'):
        dataset = tf.data.TFRecordDataset(config.test_tfrecord_path)
        dataset = dataset.take(config.num_test_examples)
        dataset = dataset.map(_decode)
        dataset = dataset.map(_image_prep)
        iterator = dataset.make_one_shot_iterator()
    return iterator, dataset


def run_training(config):
    """
        Train gaze_trainer for the given number of steps.
    """
    # train and test iterators, need dataset to create feedable iterator
    train_iterator, train_dataset = _train_iterator()
    test_iterator, _ = _test_iterator()

    # Feedable iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)
    next_element = iterator.get_next()

    # Get images and labels from iterator, create model from class
    image_batch, label_batch = next_element
    model = GazeModel(image_batch, label_batch, config)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Merge all summary ops for saving during training
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init_op)
        # Logs and model checkpoint paths defined in config
        writer = tf.summary.FileWriter(config.log_path, sess.graph)
        saver = tf.train.Saver()
        for epoch_idx in range(config.num_epochs):
            # Training
            train_handle = sess.run(train_iterator.string_handle())
            epoch_train_start = time.time()
            num_train_steps = 0
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                while True:
                    sess.run(next_element, feed_dict={handle: train_handle})
                    _, mse = sess.run([model.optimize, model.mse])
                    num_train_steps += 1
            except tf.errors.OutOfRangeError:
                epoch_train_duration = time.time() - epoch_train_start
                print('Epoch %d: Training (%.3f sec)(%d steps) - mse: %.2f' % (epoch_idx,
                                                                               epoch_train_duration,
                                                                               num_train_steps,
                                                                               mse))
                if config.save_model:
                    save_path = saver.save(sess, config.checkpoint_path)
                    print('Model checkpoint saved at %s' % save_path)
            # Testing
            test_handle = sess.run(test_iterator.string_handle())
            epoch_test_start = time.time()
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                sess.run(next_element, feed_dict={handle: test_handle})
                mse, summary = sess.run([model.mse, merged_summary_op])
                writer.add_summary(summary, epoch_idx)
            except tf.errors.OutOfRangeError:
                epoch_test_duration = time.time() - epoch_test_start
                print('Epoch %d: Testing (%.3f sec) - acc: %.2f' % (epoch_idx,
                                                                    epoch_test_duration,
                                                                    mse))
        writer.close()


def main():
    run_training(config)


if __name__ == '__main__':
    main()
