import os
from src.config.config import Config
import src.utils.data_utils as data_utils

'''
GazeConfig class contains parameters used to train the gaze models.
'''


class GazeConfig(Config):

    def __init__(self):
        # Experiment name is the root directory for logs
        self.experiment_name = 'gazedongal'
        self.build_experiment_config()

        # This config contains hyperparameters
        self.build_hyperparameter_config()

        # Seperate dataset configs
        self.train_dataset = TrainConfig()
        self.test_dataset = TestConfig()

        # Training parameters
        self.num_epochs = 30
        # Early stopping
        self.max_loss = 100
        self.best_loss = 100
        self.patience = 3
        # Save model checkpoint
        self.save_model = False
        self.save_every_n_epochs = 50
        # Optimizer parameters
        self.hyperparams['learning_rate'] = [0.01, 0.005, 0.001]
        self.hyperparams['optimizer_type'] = ['adam']

        # Model parameters
        self.model_name = 'gaze'
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

        # Generate all runs from hyperparameters
        self.generate_runs()


class TrainConfig(Config):

    def __init__(self):
        self.dataset_name = os.path.join('100118_fixedhead', 'train')
        self.dataset_type = 'gaze'
        self.tfrecord_name = 'train.tfrecords'
        # Train targets are taken from image filenames
        self.filename_regex = '(\d.\d+)_(\d.\d+).png'
        self.dataset_len = 8000
        self.shuffle = True
        self.buffer_size = 64
        self.batch_size = 16
        # Image augmentation when training
        self.image_augmentation = True
        # Brightness Augmentation
        self.random_brigtness = True
        self.brightnes_max_delta = 0.1
        # Contrast Augmentation
        self.random_contrast = True
        self.contrast_lower = 0.01
        self.contrast_upper = 0.2
        # Build the rest of the dataset related parameters
        self.build_dataset_config()
        # Create tf record dataset from data dir
        data_utils.to_tfrecords(config=self)


class TestConfig(Config):

    def __init__(self):
        self.dataset_name = os.path.join('100118_fixedhead', 'test')
        self.dataset_type = 'gaze'
        self.tfrecord_name = 'test.tfrecords'
        # Train targets are taken from image filenames
        self.filename_regex = '(\d.\d+)_(\d.\d+).png'
        self.dataset_len = 20
        self.shuffle = False
        self.buffer_size = 20
        self.batch_size = 20
        # Image augmentation when training
        self.image_augmentation = False
        # Build the rest of the dataset related parameters
        self.build_dataset_config()
        # Create tf record dataset from data dir
        data_utils.to_tfrecords(config=self)
