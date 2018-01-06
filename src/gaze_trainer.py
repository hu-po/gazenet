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
    image_shape = tf.stack([config.image_height, config.image_width, config.image_channels])
    image = tf.reshape(image, image_shape)
    return image, target


def _image_prep(image, label):
    with tf.name_scope('image prep'):
        if config.grayscale:
            image = tf.image.rgb_to_grayscale(image)
        # Apply image adjustments to reduce overfitting
        image = tf.image.random_brightness(image)
        image = tf.image.random_contrast(image)
        image = tf.image.random_hue(image)
        image = tf.image.random_saturation(image)
        # Normalize images
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label

def build_training_dataset(dataset_name, batch_size, buffer_size):
    filename = os.path.join(config.data_dir, dataset_name, 'train.tfrecords')
    with tf.name_scope('train_input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(_decode)
        dataset = dataset.map(_image_prep)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
    return dataset


def build_validation_dataset(dataset_name, batch_size, buffer_size):
    filename = os.path.join(DATA_DIR, dataset_name, 'test.tfrecords')
    with tf.name_scope('test_input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
    return dataset


def create_iterators(training_dataset, validation_dataset):
    # Datasets should both have the same structure
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    # Reinitializable iterator.
    next_element = iterator.get_next()
    # Create the initializers for both test and train datasets
    train_init_op = iterator.make_initializer(training_dataset)
    val_init_op = iterator.make_initializer(validation_dataset)
    return next_element, train_init_op, val_init_op


def run_training(dataset_name, batch_size, buffer_size, num_epochs):
    """
        Train gaze_trainer for the given number of steps.
    """

    # Training and validation datasets
    training_dataset = build_training_dataset(dataset_name, batch_size, buffer_size)
    validation_dataset = build_validation_dataset(dataset_name, batch_size, buffer_size)

    # Input images and labels for training
    next_element, train_init_op, val_init_op = create_iterators(training_dataset, validation_dataset)

    # Model requires some configs
    config = {'output_classes': OUTPUT_CLASSES,
              'learning_rate': FLAGS.learning_rate}

    # Get images and labels from iterator, create model from class
    image_batch, label_batch = next_element
    model = GazeModel(image_batch, label_batch, config)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Merge all summary ops for saving during training
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init_op)
        # Write logs to path
        log_filename = os.path.join(LOG_DIR, DATASET_NAME)
        writer = tf.summary.FileWriter(log_filename, sess.graph)
        # Write model checkpoints
        model_filename = os.path.join(MODEL_DIR, DATASET_NAME)
        saver = tf.train.Saver()
        for epoch_idx in range(num_epochs):
            # Training Step
            sess.run(train_init_op)
            epoch_train_start = time.time()
            num_train_steps = 0
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                while True:
                    sess.run(next_element)
                    _, acc = sess.run([model.optimize, model.accuracy])
                    num_train_steps += 1
            except tf.errors.OutOfRangeError:
                epoch_train_duration = time.time() - epoch_train_start
                print('Training: Epoch %d (%.3f sec) - %d steps' % (epoch_idx, epoch_train_duration, num_train_steps))
                print('        -- Accuracy %.2f ' % acc)
                # Save model
                save_path = saver.save(sess, model_filename)
                print('Model checkpoint saved at %s' % save_path)
            # Validation Step
            sess.run(val_init_op)
            epoch_val_start = time.time()
            num_val_steps = 0
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                while True:
                    sess.run(next_element)
                    acc, summary = sess.run([model.accuracy, merged_summary_op])
                    writer.add_summary(summary, epoch_idx)
                    num_val_steps += 1
            except tf.errors.OutOfRangeError:
                epoch_val_duration = time.time() - epoch_val_start
                print('Validation: Epoch %d (%.3f sec) - %d steps' % (epoch_idx, epoch_val_duration, num_val_steps))
                print('        -- Accuracy %.2f ' % acc)
        writer.close()


def main(_):
    run_training(dataset_name=FLAGS.dataset,
                 batch_size=FLAGS.batch_size,
                 buffer_size=FLAGS.buffer_size,
                 num_epochs=FLAGS.num_epochs)


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
        '--buffer_size',
        type=int,
        default=BUFFER_SIZE,
        help='Buffer size when shuffling batches'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size when running trainer.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=DATASET_NAME,
        help='Dataset name to train on.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
