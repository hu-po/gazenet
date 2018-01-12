import os
import tensorflow.contrib.slim as slim
from src.config.base_config import BaseConfig

'''
RefinerConfig class contains parameters for the refiner models
'''


class RefinerConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        'num_images',
        'tfrecord_path',
        'buffer_size',
        'batch_size'

        # Optimizer parameters
        self.initializer = slim.xavier_initializer()
        self.hyperparams['learning_rate'] = [0.01, 0.005, 0.001]
        self.hyperparams['optimizer_type'] = ['adam']

        # Resnet hyperparams
        self.hyperparams['num_rb'] = [2, 3, 4, 5]
        self.hyperparams['rb_feat'] = [8, 16, 32, 64]
        self.hyperparams['rb_kernel'] = [3, 4]
        self.hyperparams['batch_norm'] = [True, False]
