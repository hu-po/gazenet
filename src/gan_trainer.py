import os
import sys
import time
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.models.discrim_model import DiscriminatorModel
from src.models.refiner_model import RefinerModel
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
        dataset = dataset.repeat()  # Repeat dataset indefinitely
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
        dataset = dataset.repeat()  # Repeat dataset indefinitely
        dataset = dataset.shuffle(config.real_buffer_size)
        dataset = dataset.batch(config.real_batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


@train_utils.config_checker(['log_path',
                             'checkpoint_path',
                             'num_training_steps',
                             'num_refiner_steps',
                             'refiner_summary_every_n_steps',
                             'num_discrim_steps',
                             'discrim_summary_every_n_steps',
                             'save_model',
                             'save_every_n_train_steps'])
def run_training(config=None):
    # Synthetic and real image iterators
    synth_iterator, synth_batch = _synth_feed(config=config)
    real_iterator, real_batch = _real_feed(config=config)

    # Get images and labels from iterator, create model from class
    refiner_model = RefinerModel(config=config)
    discrim_model = DiscriminatorModel(config=config)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Merge all summary ops for saving during training
    refiner_summary_op = tf.summary.merge(refiner_model.summaries)
    discrim_summary_op = tf.summary.merge(discrim_model.summaries)

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init_op)
        # Logs and model checkpoint paths defined in config
        refiner_writer = tf.summary.FileWriter(config.refiner_log_path, sess.graph)
        discrim_writer = tf.summary.FileWriter(config.discrim_log_path, sess.graph)
        saver = tf.train.Saver()
        # Initalize both datasets (they repeat forever, so only need this once)
        sess.run(real_iterator.initializer)
        sess.run(synth_iterator.initializer)
        for train_step in range(config.num_training_steps):
            refiner_step_start = time.time()
            for refiner_step in range(config.num_refiner_steps):
                # Get a batch of synthetic images
                synth_image = sess.run(synth_batch)
                # Feed the synthetic images through the refiner, producing refined images
                refined_image = sess.run(refiner_model.predict,
                                         feed_dict={refiner_model.image: synth_image})
                # Feed the refined images through the discriminator to get predicted labels (fake or real?)
                pred_label = sess.run(discrim_model.predict,
                                      feed_dict={discrim_model.image: refined_image})
                # Feed the predicted labels back through the refiner model to train refiner
                # TODO: This leads to double-evaluation of the synthetic image batch, how to improve?
                _, refiner_summary = sess.run([refiner_model.optimize,
                                               refiner_summary_op],
                                              feed_dict={refiner_model.label: pred_label,
                                                         refiner_model.image: synth_image})
                if refiner_step % config.refiner_summary_every_n_steps == 0:
                    num_steps_elapsed = train_step * config.num_refiner_steps + refiner_step
                    refiner_writer.add_summary(refiner_summary, num_steps_elapsed)
            refiner_step_duration = time.time() - refiner_step_start
            discrim_step_start = time.time()
            for discrim_step in range(config.num_discrim_steps):
                synth_image = sess.run(synth_batch)
                # Feed synthetic images through refiner network
                refined_image = sess.run(refiner_model.predict, feed_dict={refiner_model.image: synth_image})
                real_image = sess.run(real_batch)
                mixed_batch = sess.run(discrim_model.mixed_image_batch,
                                       feed_dict={discrim_model.real_image: real_image,
                                                  discrim_model.refined_image: refined_image})
                mixed_image = mixed_batch[0]
                mixed_label = mixed_batch[1]
                # Train discriminator network using mixed images
                _, discrim_summary = sess.run([discrim_model.optimize,
                                               discrim_summary_op],
                                              feed_dict={discrim_model.label: mixed_label,
                                                         discrim_model.image: mixed_image})
                if discrim_step % config.discrim_summary_every_n_steps == 0:
                    num_steps_elapsed = train_step * config.num_discrim_steps + discrim_step
                    discrim_writer.add_summary(discrim_summary, num_steps_elapsed)
            discrim_step_duration = time.time() - discrim_step_start
            print('Step %d : refiner (%.3f sec) and discriminator (%.3f sec)' % (train_step,
                                                                                 refiner_step_duration,
                                                                                 discrim_step_duration))
        # Close both writers
        discrim_writer.close()
        refiner_writer.close()


def main():
    config = GANConfig()
    base_utils.image_to_tfrecords(config=config)
    run_training(config=config)


if __name__ == '__main__':
    main()
