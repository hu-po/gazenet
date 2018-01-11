import os
import datetime
import random
import itertools
from collections import OrderedDict

'''
The base config class is extended to create all other config classes
'''


class BaseConfig(object):

    def __init__(self, experiment_name='mystery'):
        # Root directory for entire repo
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # Local directories contain logs, datasets, saved model
        self.data_dir = os.path.join(self.root_dir, 'local', 'data')
        self.log_dir = os.path.join(self.root_dir, 'local', 'logs')
        self.model_dir = os.path.join(self.root_dir, 'local', 'models')

        # Create experiment specific log and checkpoint directories
        d = datetime.datetime.today()
        experiment_name = '%s_%sm_%sd_%shr_%smin' % (experiment_name, d.month, d.day, d.hour, d.minute)
        self.log_path = os.path.join(self.log_dir, experiment_name)
        self.make_path(self.log_path)
        self.checkpoint_path = os.path.join(self.model_dir, experiment_name)
        self.make_path(self.checkpoint_path)
        print('Created log and checkpoint directories for experiment %s' % experiment_name)

        # Each run within an experiment will have a run-specific name
        self.run_specific_name = ''
        self.run_log_path = None
        self.run_checkpoint_path = None

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

        self.runs = []


    @staticmethod
    def make_path(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def prepare_experiment(self):
        # Generate all runs (all possible permutations of hyperparameters)
        self._generate_runs()
        self.num_runs = len(self.runs)

    def prepare_run(self, idx):
        self._set_hyperparams(idx)
        self._create_run_directories()

    def _generate_runs(self):
        permutation_builder = []
        for key, value in self.hyperparams.items():
            permutation_builder.append(range(len(value)))
        # Get all possible permutations from the permutations builder
        permutations = list(itertools.product(*permutation_builder))
        permutations = [list(a) for a in permutations]
        # Shuffle prevents it from being a grid search
        random.shuffle(permutations)
        # Set class property
        self.runs = permutations

    def _set_hyperparams(self, idx):
        permutation = self.runs[idx]
        self.run_hyperparams = OrderedDict()
        for i, key in enumerate(self.hyperparams.keys()):
            value = self.hyperparams[key][permutation[i]]
            self.run_hyperparams[key] = value
            setattr(self, key, value)

    def _create_run_directories(self):
        for key, value in self.run_hyperparams.items():
            str_value = str(value)
            if isinstance(value, list):
                str_value = '_'.join(str(a) for a in value)
            self.run_specific_name += '_%s_%s' % (key, str_value)
        self.run_log_path = os.path.join(self.log_path, self.run_specific_name)
        self.make_path(self.log_path)
        self.run_checkpoint_path = os.path.join(self.checkpoint_path, self.run_specific_name)
        self.make_path(self.checkpoint_path)
