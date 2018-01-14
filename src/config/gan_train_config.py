import os
import sys
import tensorflow.contrib.slim as slim

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.config.config import Config
import src.utils.config_utils as config_util

'''
This file contains all the parameters for training the GAN component of this project. Each
inherits from whatever config parent class is most relevant

GANConfig - Experiment

'''


class GANConfig(Config):
    experiment_name = 'dualgan'

    # Training parameters (from Algorithm 1 in [1])
    num_training_steps = 5  # 100  # T
    num_refiner_steps = 5  # 200  # Kg
    num_discrim_steps = 5  # 50  # Kd

    def __init__(self):
        build_experiment_config()

        # Seperate dataset configs
        real_dataset = RealConfig()
        fake_dataset = FakeConfig()

        # Configs for both of the models
        refiner_model = RefinerConfig(exp_config_handle=self)
        discrim_model = DiscrimConfig(exp_config_handle=self)


class DiscrimConfig(Config):

    def __init__(self, exp_config_handle=None):
        self.model_name = 'discrim'
        # This config contains hyperparameters
        self.build_hyperparameter_config(exp_config_handle=exp_config_handle)
        # Log saving every n steps
        self.save_logs = True
        self.summary_every_n_steps = 5
        # Save model checkpoint
        self.save_model = False
        self.save_every_n_train_steps = 50
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
        self.generate_runs()


class RefinerConfig(Config):

    def __init__(self, exp_config_handle=None):
        self.model_name = 'refiner'
        # This config contains hyperparameters
        self.build_hyperparameter_config(exp_config_handle=exp_config_handle)
        # Log saving every n steps
        self.save_logs = True
        self.summary_every_n_steps = 5
        # Save model checkpoint
        self.save_model = False
        self.save_every_n_train_steps = 50
        # Loss parameters
        self.regularization_lambda = 1
        # Optimizer parameters
        self.initializer = slim.xavier_initializer()
        self.hyperparams['learning_rate'] = [0.01, 0.005, 0.001]
        self.hyperparams['optimizer_type'] = ['adam']
        # Resnet hyperparams
        self.hyperparams['num_rb'] = [2, 3, 4, 5]
        self.hyperparams['rb_feat'] = [8, 16, 32, 64]
        self.hyperparams['rb_kernel'] = [3, 4]
        self.hyperparams['batch_norm'] = [True, False]
        # Generate all runs from hyperparameters
        self.generate_runs()
