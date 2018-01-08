import os
from src.config.base_config import BaseConfig

'''
GazeConfig class contains parameters used to train the gaze models.
'''


class GazeConfig(BaseConfig):

    def __init__(self):
        super().__init__(run_name='gaze')
        # Gazenet uses a single dataset
        self.dataset_name = '04012018_headlook'
        # Train targets are taken from image filenames
        self.filename_regex = '(\d.\d+)_(\d.\d+).png'
        self.train_test_split = 0.95
        # Brightness Augmentation
        self.random_brigtness = True
        self.brightnes_max_delta = 0.1
        # Contrast Augmentation
        self.random_contrast = True
        self.contrast_lower = 0.01
        self.contrast_upper = 0.2

        # Training parameters
        self.num_epochs = 400
        self.batch_size = 4
        # Number of train examples (if you want to limit training data)
        self.num_train_examples = 200
        # Number of test examples in each validation step
        self.num_test_examples = 16
        # Bigger buffer means better shuffling but slower start up and more memory used.
        self.buffer_size = 10
        # Save model checkpoint
        self.save_model = True
        self.save_every_n_epochs = 50
        # Model dropout
        self.dropout_keep_prob = 0.8
        # Optimizer parameters
        self.learning_rate = 0.01
        self.optimizer_type = 'rmsprop'

        # Dataset and tfrecord paths
        self.dataset_path = os.path.join(self.data_dir, self.dataset_name)
        self.train_tfrecord_path = os.path.join(self.dataset_path, 'train.tfrecords')
        self.test_tfrecord_path = os.path.join(self.dataset_path, 'test.tfrecords')

        # Set up seperate train and test log directories
        self.train_log_path = os.path.join(self.log_path, 'train')
        self.test_log_path = os.path.join(self.log_path, 'test')
        self.make_path(self.train_log_path)
        self.make_path(self.test_log_path)