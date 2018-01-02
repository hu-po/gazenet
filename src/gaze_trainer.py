import tensorflow as tf
import os, sys
import argparse
import time
from src.gaze_model import GazeModel
from src.config import DATA_DIR

'''
This file is used to train the gaze net. It contains functions for reading
and decoding the data (should be in TFRecords format).

Much of this is sourced from the TF mnist example:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
'''

# Default Parameters:
DATASET_NAME = '020118_fingers'
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.01

# Model parameters
OUTPUT_CLASSES = 4


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    # Extract image dimmensions from features
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # Extract label
    label = tf.cast(features['label'], tf.int32)

    # Extract image from image string (convert to uint8 and then re-size)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.pack([height, width, 3])
    image = tf.reshape(image, image_shape)

    return image, label


def augment(image, label):
    # Flip image from left to right
    image = tf.image.random_flip_left_right(image)

    return image, label


def normalize(image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image, label


def inputs(train, batch_size, num_epochs):
    """
    Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, num_pixels]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label
      This function creates a one_shot_iterator, meaning that it will only iterate
      over the dataset once. On the other hand there is no special initialization
      required.
    """
    if not num_epochs:
        num_epochs = None
    filename = os.path.join(DATA_DIR,
                            FLAGS.dataset,
                            'train.tfrecords' if train else 'test.tfrecords')

    with tf.name_scope('input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)
        dataset = dataset.map(augment)
        dataset = dataset.map(normalize)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more memory used.
        min_after_dequeue = 10

        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        capacity = min_after_dequeue + 3 * batch_size

        dataset = dataset.shuffle(capacity)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def run_training():
    """
        Train gaze_trainer for the given number of steps.
    """

    # Input images and labels.
    image_batch, label_batch = inputs(train=True,
                                      batch_size=FLAGS.batch_size,
                                      num_epochs=FLAGS.num_epochs)

    # Model requires some configs
    config = {'output_class': OUTPUT_CLASSES,
              'learning_rate': FLAGS.learning_rate}

    # Model from class
    model = GazeModel(image_batch, label_batch, config)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session as sess:
        # Initialize variables
        sess.run(init_op)

        try:
            step = 0
            while True:  # train until OutOfRangeError
                start_time = time.time()

                # Run one step of the model.
                loss, error = sess.run([model.optimize, model.error])

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d (%.3f sec): loss = %.2f, error = %.2f ' % (step,
                                                                              loss,
                                                                              error,
                                                                              duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=LEARNING_RATE,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=NUM_EPOCHS,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size when running trainer.'
    )
    parser.add_argument(
        '--dataset',
        type=int,
        default=DATASET_NAME,
        help='Dataset name to train on.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
