import os

'''
The base config class is extended to create all other config classes
'''


class BaseConfig(object):

    def __init__(self):
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
