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
        self.real_dataset_name = '080118_real'
        # Clip the dataset sizes
        self.num_synth_images = 1000
        self.num_real_images = 100

        # Training parameters (from Algorithm 1 in [1])
        self.num_training_steps = 100  # T
        self.num_refiner_steps = 200  # Kg
        self.num_discrim_steps = 50  # Kd
        self.synth_batch_size = 8
        self.real_batch_size = 8
        # Bigger buffer means better shuffling but slower start up and more memory used.
        self.synth_buffer_size = 16
        self.real_buffer_size = 16
        # Log saving every n steps
        self.refiner_summary_every_n_steps = 50
        self.discrim_summary_every_n_steps = 10
        # Save model checkpoint
        self.save_model = False
        self.save_every_n_train_steps = 50

        # Refiner Model
        self.refiner_initializer = slim.xavier_initializer()
        self.regularization_lambda = 1
        self.num_resnet_blocks = 4
        self.num_feat_per_resnet_block = 64
        self.kernel_size_resnet_block = 3
        self.refiner_learning_rate = 0.01
        self.refiner_optimizer_type = 'rmsprop'

        # Discriminator
        self.discrim_initializer = slim.xavier_initializer()
        self.discrim_learning_rate = 0.01
        self.discrim_optimizer_type = 'rmsprop'

        # Dataset and tfrecord paths
        self.synth_dataset_path = os.path.join(self.data_dir, self.synth_dataset_name)
        self.real_dataset_path = os.path.join(self.data_dir, self.real_dataset_name)
        self.synth_tfrecord_path = os.path.join(self.synth_dataset_path, 'image.tfrecords')
        self.real_tfrecord_path = os.path.join(self.real_dataset_path, 'image.tfrecords')

        # Set up seperate refiner and discriminator log directories
        self.refiner_log_path = os.path.join(self.log_path, 'refiner')
        self.discrim_log_path = os.path.join(self.log_path, 'discrim')
        self.make_path(self.refiner_log_path)
        self.make_path(self.discrim_log_path)