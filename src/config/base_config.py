import os
import datetime
import random
import itertools
from collections import OrderedDict

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
        self.image_channels_input = 3
        # Grayscale images are quicker, and depending on problem color is not important
        self.grayscale = True
        if self.grayscale:
            self.image_channels = 1

        # Dictionary of all hyperparameter values
        self.hyperparams = OrderedDict()
        self.run_idx = 0
        self.runs = []

    @staticmethod
    def make_path(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def permute_hyperparams(self):
        permutation_builder = []
        for key, value in self.hyperparams.items():
            permutation_builder.append(range(len(value)))
        # Get all possible perumations from the permuation builder
        permutations = list(itertools.product(*permutation_builder))
        permutations = [list(a) for a in permutations]
        # Shuffle for trying out more different combinations
        random.shuffle(permutations)
        # Set class property
        self.runs = permutations

    def set_hyperparams(self):
        permutation = self.runs[self.run_idx]
        self.run_hyperparams = OrderedDict()
        for i, key in enumerate(self.hyperparams.keys()):
            value = self.hyperparams[key][permutation[i]]
            self.run_hyperparams[key] = value
            setattr(self, key, value)
        self.run_idx += 1

    def create_run_directories(self):
        d = datetime.datetime.today()
        run_specific_name = '%s_%sm_%sd_%shr_%smin' % (self.net_name, d.month, d.day, d.hour, d.minute)
        for key, value in self.run_hyperparams.items():
            run_specific_name += '_%s_%s' % (key, str(value))
        self.log_path = os.path.join(self.log_dir, run_specific_name)
        self.checkpoint_path = os.path.join(self.model_dir, run_specific_name)
        self.make_path(self.log_path)
        self.make_path(self.checkpoint_path)
        print('Creating run directories for %s' % run_specific_name)
