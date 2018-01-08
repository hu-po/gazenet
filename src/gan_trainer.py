import os
import sys
import time
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.discriminator_model import DiscriminatorModel
from src.refiner_model import RefinerModel
from src.config.gan_config import GANConfig
import src.utils.train_utils as train_utils
import src.utils.base_utils as base_utils

'''
This file is used to train the GAN, which is composed of a refiner net and a 
 discriminator net. It follows Algorithm 1 in [1].
'''


@train_utils.config_checker(['num_synth_images', 'synth_tfrecord_path',
                             'synth_buffer_size', 'synth_batch_size'])
def _synth_feed(config=None):
    with tf.name_scope('synth_input'):
        dataset = tf.data.TFRecordDataset(config.synth_tfrecord_path)
        dataset = dataset.take(config.num_synth_images)
        dataset = dataset.map(lambda x: train_utils.decode_image(x, config=config))
        dataset = dataset.map(lambda i: train_utils.grayscale(i, config=config))
        dataset = dataset.map(lambda i: train_utils.standardize(i, config=config))
        dataset = dataset.repeat() # Repeat dataset indefinitely
        dataset = dataset.shuffle(config.synth_buffer_size)
        dataset = dataset.batch(config.synth_batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@train_utils.config_checker(['num_real_images', 'real_tfrecord_path',
                             'real_buffer_size', 'real_batch_size'])
def _real_feed(config=None):
    with tf.name_scope('real_input'):
        dataset = tf.data.TFRecordDataset(config.real_tfrecord_path)
        dataset = dataset.take(config.num_real_images)
        dataset = dataset.map(lambda x: train_utils.decode_image(x, config=config))
        dataset = dataset.map(lambda i: train_utils.grayscale(i, config=config))
        dataset = dataset.map(lambda i: train_utils.standardize(i, config=config))
        dataset = dataset.repeat() # Repeat dataset indefinitely
        dataset = dataset.shuffle(config.real_buffer_size)
        dataset = dataset.batch(config.real_batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@train_utils.config_checker(['log_path',
                             'checkpoint_path',
                             'num_training_steps',
                             'num_refiner_steps',
                             'num_discriminator_steps',
                             'save_model',
                             'save_every_n_train_steps'])
def run_training(config=None):
    # Synthetic and real image iterators
    synth_iterator, synth_batch = _synth_feed(config=config)
    real_iterator, real_batch = _real_feed(config=config)

    # Get images and labels from iterator, create model from class
    refiner_model = RefinerModel(config=config)
    discriminator_model = DiscriminatorModel(config=config)

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
        # Initalize both datasets (they repeat forever, so only need this once)
        sess.run(real_iterator.initializer)
        sess.run(synth_iterator.initializer)
        for train_step in range(config.num_training_steps):
            refiner_step_start = time.time()
            for refiner_step in range(config.num_refiner_steps):
                synth_image_batch = sess.run(synth_batch)

            refiner_step_duration = time.time() - refiner_step_start


            discriminator_step_start = time.time()
            for discriminator_step in range(config.num_discriminator_steps):
                synth_image_batch = sess.run(synth_batch)
                real_image_batch = sess.run(real_batch)

            discriminator_step_duration = time.time() - discriminator_step_start
            print('Step %d : refiner (%.3f sec) and discriminator (%.3f sec)' % (train_step,
                                                                                 refiner_step_duration,
                                                                                 discriminator_step_duration))
            writer.add_summary(summary, train_step)
        writer.close()


def main():
    config = GANConfig()
    base_utils.image_to_tfrecords(config=config)
    run_training(config=config)


if __name__ == '__main__':
    main()
