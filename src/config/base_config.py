import os
import datetime

'''
The base config class is extended to create all other config classes
'''


class BaseConfig(object):

    def __init__(self, run_name):
        # Root directory for entire repo
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # Local directories contain logs, datasets, saved model
        self.data_dir = os.path.join(self.root_dir, 'local', 'data')
        self.log_dir = os.path.join(self.root_dir, 'local', 'logs')
        self.model_dir = os.path.join(self.root_dir, 'local', 'models')

        # Images dimmensions
        self.image_width = 128
        self.image_height = 96
        self.image_channels = 3
        # Grayscale images are quicker, and depending on problem color is not important
        self.grayscale = True

        # Set up run-specific checkpoint and log paths
        d = datetime.datetime.today()
        run_specific_name = '%s_%s_%s_%s_%s' % (run_name, d.month, d.day, d.hour, d.minute)
        self.log_path = os.path.join(self.log_dir, run_specific_name)
        self.checkpoint_path = os.path.join(self.log_dir, run_specific_name)
        os.mkdir(self.log_path)
        os.mkdir(self.checkpoint_path)