import os
from src.config.base_config import BaseConfig

'''
GazeConfig class contains parameters used to train the gaze models.
'''


class GazeConfig(BaseConfig):

    def __init__(self):
        super().__init__()
        # Net name is used to identify logs
        self.net_name = 'gaze'
        # Gazenet uses a single dataset
        self.dataset_name = '100118_fixedhead'
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
        self.num_epochs = 1
        self.batch_size = 16
        # Number of train examples (if you want to limit training data)
        self.num_train_examples = 8000
        # Number of test examples in each validation step
        self.num_test_examples = 16
        # Bigger buffer means better shuffling but slower start up and more memory used.
        self.buffer_size = 32
        # Save model checkpoint
        self.save_model = False
        self.save_every_n_epochs = 50
        # Optimizer parameters
        self.hyperparams['learning_rate'] = [0.01, 0.005, 0.001]
        self.hyperparams['optimizer_type'] = ['rmsprop', 'sgd', 'adam']

        # Model parameters
        # self.hyperparams['dropout_keep_prob'] = [0.6]
        # self.hyperparams['num_conv_layers_1'] = [2, 3]
        # self.hyperparams['num_feature_1'] = [32, 64]
        # self.hyperparams['kernel_1'] = [3, 4]
        # self.hyperparams['max_pool_1'] = [True]
        # self.hyperparams['num_conv_layers_2'] = [1, 2]
        # self.hyperparams['num_feature_2'] = [64, 128]
        # self.hyperparams['kernel_2'] = [3, 4]
        # self.hyperparams['max_pool_2'] = [True]
        # self.hyperparams['num_fc_layers'] = [1, 2]
        # self.hyperparams['fc_layer_num'] = [32, 64, 128]

        self.dropout_keep_prob = 0.6
        self.hyperparams['num_conv_layers_1'] = [2, 3]
        self.hyperparams['num_feature_1'] = [32, 64]
        self.hyperparams['kernel_1'] = [3, 4]
        self.max_pool_1 = True
        self.hyperparams['num_conv_layers_2'] = [1, 2]
        self.hyperparams['num_feature_2'] = [64, 128]
        self.hyperparams['kernel_2'] = [3, 4]
        self.max_pool_2 = True
        self.hyperparams['num_fc_layers'] = [1, 2]
        self.hyperparams['fc_layer_num'] = [32, 64, 128]

        # Dataset and tfrecord paths
        self.dataset_path = os.path.join(self.data_dir, self.dataset_name)
        self.train_tfrecord_path = os.path.join(self.dataset_path, 'train.tfrecords')
        self.test_tfrecord_path = os.path.join(self.dataset_path, 'test.tfrecords')

    def create_run_directories(self):
        super().create_run_directories()
        self.train_log_path = os.path.join(self.log_path, 'train')
        self.test_log_path = os.path.join(self.log_path, 'test')
        self.make_path(self.train_log_path)
        self.make_path(self.test_log_path)
