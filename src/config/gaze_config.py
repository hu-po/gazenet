import os
from src.config.config import Config

'''
GazeConfig class contains parameters used to train the gaze models.
'''


class GazeConfig(Config):

    def __init__(self):
        # Experiment name is the root directory for logs
        self.experiment_name = 'res_batch_gaze'
        self.build_experiment_config()

        # This config contains hyperparameters
        self.build_hyperparameter_config()

        # Gazenet uses a single dataset
        self.dataset_name = '100118_fixedhead'
        # Train targets are taken from image filenames
        self.filename_regex = '(\d.\d+)_(\d.\d+).png'
        self.train_test_split = 0.95
        # Brightness Augmentation
        self.random_brigtness = False
        self.brightnes_max_delta = 0.1
        # Contrast Augmentation
        self.random_contrast = False
        self.contrast_lower = 0.01
        self.contrast_upper = 0.2

        # Training parameters
        self.num_epochs = 30
        self.batch_size = 16
        # Early stopping
        self.patience = 3
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
        self.hyperparams['optimizer_type'] = ['adam']

        # Model parameters
        self.dropout_keep_prob = 0.6
        self.hyperparams['fc_layers'] = [[128, 128, 64],
                                         [256, 32],
                                         [64, 64],
                                         [256, 64],
                                         [128, 32]]
        self.hyperparams['dimred_feat'] = [32, 64, 128]
        self.hyperparams['dimred_kernel'] = [4, 6, 8]
        self.hyperparams['dimred_stride'] = [2, 4]
        # Resnet hyperparams
        self.hyperparams['num_rb'] = [2, 3, 4, 5]
        self.hyperparams['rb_feat'] = [8, 16, 32, 64]
        self.hyperparams['rb_kernel'] = [3, 4]
        self.hyperparams['batch_norm'] = [True, False]

        # Dataset and tfrecord paths
        self.dataset_path = os.path.join(self.data_dir, self.dataset_name)
        self.train_tfrecord_path = os.path.join(self.dataset_path, 'train.tfrecords')
        self.test_tfrecord_path = os.path.join(self.dataset_path, 'test.tfrecords')
