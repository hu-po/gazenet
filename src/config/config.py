import os
import sys

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.utils.data_utils as data_utils
import src.utils.config_utils as config_util

'''
The base config classes are extended to create all other config classes
'''


class Config(object):
    # Root directory for entire repo
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Local directories contain logs, datasets, saved model
    data_dir = os.path.join(root_dir, 'local', 'data')
    log_dir = os.path.join(root_dir, 'local', 'logs')
    model_dir = os.path.join(root_dir, 'local', 'models')
    # Images dimensions
    image_width = 128
    image_height = 96
    image_channels = 1

class ModelConfig(Config):
    def __init__(self):
        # Build the rest of the dataset related parameters
        self.dataset_path = os.path.join(self.data_dir, self.dataset_name)
        self.tfrecord_path = os.path.join(self.dataset_path, self.tfrecord_name)
        # Create tf record dataset from data dir
        data_utils.to_tfrecords(config=self)
