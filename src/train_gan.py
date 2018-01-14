import os
import sys
import time
import random
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.config.config import config_checker
from src.models.discrim_model import DiscriminatorModel
from src.models.refiner_model import RefinerModel
from src.config.gan_train_config import GANConfig
import src.utils.train_utils as train_utils

'''
This file is used to train the GAN, which is composed of a refiner net and a 
 discriminator net. It follows Algorithm 1 in [1].
'''


@config_checker(['real_dataset',
                 'fake_dataset'])
def refiner_solo_train(config=None):
    pass


@config_checker(['real_dataset',
                 'fake_dataset',
                 'num_training_steps',
                 'num_refiner_steps',
                 'num_discrim_steps'])
def adversarial_training(config=None):
    # Synthetic and real image iterators
    synth_iterator, synth_batch_op = train_utils.image_feed(config=config.real_dataset)
    real_iterator, real_batch_op = train_utils.image_feed(config=config.fake_dataset)

    # Build models
    refiner_model = RefinerModel(config=config.refiner_model)
    discrim_model = DiscriminatorModel(config=config.discrim_model)

    # Mixed batch to be fed to the discriminator during training
    real_images, refined_images, mixed_batch_op = train_utils.mixed_image_batch(config=config)

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
        refiner_writer = tf.summary.FileWriter(os.path.join(config.refiner_model.run_log_path, 'refiner'), sess.graph)
        discrim_writer = tf.summary.FileWriter(os.path.join(config.discrim_model.run_log_path, 'discrim'), sess.graph)
        # Initalize both datasets (they repeat forever, so only need this once)
        sess.run(real_iterator.initializer)
        sess.run(synth_iterator.initializer)
        for train_step in range(config.num_training_steps):
            refiner_step_start = time.time()
            for refiner_step in range(config.num_refiner_steps):
                # Get a batch of synthetic images
                synth_image = sess.run(synth_batch_op)
                # Feed the synthetic images through the refiner, producing refined images
                refined_image = sess.run(refiner_model.predict, feed_dict={refiner_model.image: synth_image,
                                                                           refiner_model.is_training: False})
                # Feed the refined images through the discriminator to get probability of being fake or real
                pred_label = sess.run(discrim_model.predict, feed_dict={discrim_model.image: refined_image,
                                                                        discrim_model.is_training: False})
                # Feed this probability label back through the refiner model to train refiner
                # TODO: This leads to double-evaluation of the synthetic image batch, how to improve?
                _, refiner_summary = sess.run([refiner_model.optimize,
                                               refiner_summary_op],
                                              feed_dict={refiner_model.label: pred_label,
                                                         refiner_model.image: synth_image,
                                                         refiner_model.is_training: True})
                if refiner_step % config.refiner_model.summary_every_n_steps == 0:
                    num_steps_elapsed = train_step * config.num_refiner_steps + refiner_step
                    refiner_writer.add_summary(refiner_summary, num_steps_elapsed)
            refiner_step_duration = time.time() - refiner_step_start
            discrim_step_start = time.time()
            for discrim_step in range(config.num_discrim_steps):
                synth_batch = sess.run(synth_batch_op)
                real_batch = sess.run(real_batch_op)
                # Feed synthetic images through refiner network
                refined_batch = sess.run(refiner_model.predict, feed_dict={refiner_model.image: synth_batch,
                                                                           refiner_model.is_training: False})
                # Get a batch of mixed refined and real images
                mixed_image, mixed_label = sess.run(mixed_batch_op, feed_dict={real_images: real_batch,
                                                                               refined_images: refined_batch})
                # Train discriminator network using mixed images
                _, discrim_summary = sess.run([discrim_model.optimize,
                                               discrim_summary_op],
                                              feed_dict={discrim_model.label: mixed_label,
                                                         discrim_model.image: mixed_image,
                                                         discrim_model.is_training: True})
                if discrim_step % config.discrim_model.summary_every_n_steps == 0:
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
    # Pick random runs for now
    refiner_run_idx = random.randint(0, config.refiner_model.num_runs)
    discrim_run_idx = random.randint(0, config.discrim_model.num_runs)
    # Set up specific runs for both models
    config.refiner_model.prepare_run(refiner_run_idx)
    config.discrim_model.prepare_run(discrim_run_idx)
    adversarial_training(config=config)


if __name__ == '__main__':
    main()
