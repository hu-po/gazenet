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
        for train_idx in range(config.num_training_steps):





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
    base_utils.gazedata_to_tfrecord(config=config)
    run_training(config=config)


if __name__ == '__main__':
    main()
