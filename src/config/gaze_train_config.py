import sys
import os
from src.config.config import Config
import src.utils.data_utils as data_utils
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.config_utils as config_util

'''
GazeConfig class contains parameters used to train the gaze models.
'''


class GazeConfig(Config):
    # Experiment name is the root directory for logs
    experiment_name = 'gazedongal'
    # Training parameters
    num_epochs = 30
    # Early stopping
    max_loss = 100
    best_loss = 100
    patience = 3

    def __init__(self):
        config_util.build_experiment_config(self)
        # This config contains hyperparameters
        config_util.build_hyperparameter_config(self)
        # Separate configs for model and datasets
        self.train_dataset = TrainConfig()
        self.test_dataset = TestConfig()
        self.gaze_model = GazeModel()


class GazeModel(Config):

    def __init__(self):
        self.model_name = 'gaze'
        # Optimizer parameters
        self.initializer = slim.xavier_initializer()
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
        # Generate all runs from hyperparameters
        config_util.generate_runs()

