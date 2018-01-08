import os
import tensorflow.contrib.slim as slim
from src.config.base_config import BaseConfig

'''
GANConfig class contains parameters used to train the (refiner and discriminator nets)
'''


class GANConfig(BaseConfig):

    def __init__(self):
        super().__init__(run_name='gan')
        # GAN needs two input datasets and will create an output dataset
        self.synth_dataset_name = '04012018_headlook'
        self.real_dataset_name = '070118_real'
        # Clip the dataset sizes
        self.num_synth_images = 5000
        self.num_real_images = 100

        # Training parameters (from Algorithm 1 in [1])
        self.num_training_steps = 400  # T
        self.num_refiner_steps = 50  # Kg
        self.num_discriminator_steps = 1  # Kd
        self.synth_batch_size = 16
        self.real_batch_size = 16
        # Bigger buffer means better shuffling but slower start up and more memory used.
        self.synth_buffer_size = 100
        self.real_buffer_size = 100
        # Save model checkpoint
        self.save_model = True
        self.save_every_n_steps = 50

        # Refiner Model
        self.refiner_initializer = slim.xavier_initializer()
        self.regularization_lambda = 1
        self.refiner_learning_rate = 0.01
        self.refiner_optimizer_type = 'rmsprop'

        # Discriminator
        self.discriminator_initializer = slim.xavier_initializer()
        self.discriminator_learning_rate = 0.01
        self.discriminator_optimizer_type = 'rmsprop'

        # Dataset and tfrecord paths
        self.synth_dataset_path = os.path.join(self.data_dir, self.synth_dataset_name)
        self.real_dataset_path = os.path.join(self.data_dir, self.real_dataset_name)
        self.synth_tfrecord_path = os.path.join(self.synth_dataset_path, 'image.tfrecords')
        self.real_tfrecord_path = os.path.join(self.real_dataset_path, 'image.tfrecords')
