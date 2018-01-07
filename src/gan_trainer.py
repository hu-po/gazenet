import os
import sys
import time
import datetime
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.discriminator_model import DiscriminatorModel
from src.refiner_model import RefinerModel
import src.utils.train_utils as train_utils

'''
This file is used to train the GAN, which is composed of a refiner net and a 
 discriminator net. It follows Algorithm 1 in [1].
'''

@train_utils.config_checker(['num_train_examples',
                             'train_tfrecord_path',
                             'buffer_size',
                             'batch_size'])
def _train_feed(config=None):
    with tf.name_scope('train_input'):
        dataset = tf.data.TFRecordDataset(config.train_tfrecord_path)
        dataset = dataset.take(config.num_train_examples)
        dataset = dataset.map(lambda x: train_utils.decode_gaze(x, config=config))
        dataset = dataset.map(lambda i, t: train_utils.image_augmentation(i, t, config=config))
        dataset = dataset.map(lambda i, t: train_utils.grayscale(i, t, config=config))
        dataset = dataset.map(lambda i, t: train_utils.standardize(i, t, config=config))
        dataset = dataset.shuffle(config.buffer_size)
        dataset = dataset.batch(config.batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@train_utils.config_checker(['num_test_examples',
                             'test_tfrecord_path',
                             'buffer_size',
                             'batch_size'])
def _test_feed(config=None):
    with tf.name_scope('test_input'):
        dataset = tf.data.TFRecordDataset(config.test_tfrecord_path)
        dataset = dataset.take(config.num_test_examples)
        dataset = dataset.map(lambda x: train_utils.decode_gaze(x, config=config))
        dataset = dataset.map(lambda i, t: train_utils.grayscale(i, t, config=config))
        dataset = dataset.map(lambda i, t: train_utils.standardize(i, t, config=config))
        dataset = dataset.batch(config.num_test_examples)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@train_utils.config_checker(['log_path',
                             'checkpoint_path',
                             'num_epochs',
                             'save_model',
                             'save_every_n_epochs',
                             'num_epochs'])
def run_training(config=None):
    """
        Train gaze_trainer for the given number of steps.
    """
    # train and test iterators, need dataset to create feedable iterator
    train_iterator, train_batch = _train_feed(config=config)
    test_iterator, test_batch = _test_feed(config=config)

    # Get images and labels from iterator, create model from class
    model = GazeModel(config)

    # Set up run-specific checkpoint and log paths
    d = datetime.datetime.today()
    run_specific_name = '%s_%s_%s_%s' % (d.month, d.day, d.hour, d.minute)
    log_path = os.path.join(config.log_path, run_specific_name)
    checkpoint_path = os.path.join(config.checkpoint_path, run_specific_name)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Merge all summary ops for saving during training
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init_op)
        # Logs and model checkpoint paths defined in config
        # TODO: change log path every run to keep historical run data
        writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
        for epoch_idx in range(config.num_epochs):
            # Training
            epoch_train_start = time.time()
            num_train_steps = 0
            sess.run(train_iterator.initializer)
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                while True:
                    image_batch, label_batch = sess.run(train_batch)
                    _, _, mse = sess.run([model.optimize,
                                          model.mse,
                                          model.train_loss], feed_dict={model.image: image_batch,
                                                                        model.label: label_batch,
                                                                        model.train_mode: True})
                    num_train_steps += 1
            except tf.errors.OutOfRangeError:
                epoch_train_duration = time.time() - epoch_train_start
                print('Epoch %d: Training (%.3f sec)(%d steps) - mse: %.2f' % (epoch_idx,
                                                                               epoch_train_duration,
                                                                               num_train_steps,
                                                                               mse))
                if config.save_model and ((epoch_idx + 1) % config.save_every_n_epochs) == 0:
                    os.mkdir(checkpoint_path)
                    save_path = saver.save(sess, os.path.join(checkpoint_path, str(epoch_idx + 1)))
                    print('Model checkpoint saved at %s' % save_path)
            # Testing
            epoch_test_start = time.time()
            sess.run(test_iterator.initializer)
            try:
                image_batch, label_batch = sess.run(test_batch)
                _, mse, summary = sess.run([model.mse,
                                            model.test_loss,
                                            merged_summary_op], feed_dict={model.image: image_batch,
                                                                           model.label: label_batch,
                                                                           model.train_mode: False})
            except tf.errors.OutOfRangeError:
                epoch_test_duration = time.time() - epoch_test_start
                print('Epoch %d: Testing (%.3f sec) - acc: %.2f' % (epoch_idx,
                                                                    epoch_test_duration,
                                                                    mse))
            writer.add_summary(summary, (epoch_idx + 1))
        writer.close()


def main():
    import src.gan_config as config
    run_training(config=config)


if __name__ == '__main__':
    main()
