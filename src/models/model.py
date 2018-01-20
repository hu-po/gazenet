import os
import sys
import random
import itertools
from collections import OrderedDict

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import Config


class Model(Config):

    def __init__(self):
        # Convert hyperparameters into an ordered dict
        list_of_tuples = [(key, self.hyperparams[key]) for key in self.hyperparams.keys()]
        self.hyperparams = OrderedDict(list_of_tuples)

        # Each 'run' is a particular permutation of hyperparameters
        self.run_idx = 0
        self.runs = self._generate_runs()

    def next_run(self):
        permutation = self.runs[self.run_idx]
        run_specific_name = self.model_name
        for i, key in enumerate(self.hyperparams.keys()):
            # Change the hyperparam class property
            value = self.hyperparams[key][permutation[i]]
            setattr(self, key, value)
            # Add the hyperparam to the run_specifc name
            str_value = str(value)
            if isinstance(value, list):
                str_value = '_'.join(str(a) for a in value)
            run_specific_name += '_%s_%s' % (key, str_value)
        self.run_idx += 1
        return run_specific_name

    def build_model_params(self):
        model_params = {}
        for param_name in self.model_params:
            value = getattr(self, param_name, None)
            model_params[param_name] = value
        return model_params

    def _generate_runs(self):
        # Generate all runs (all possible permutations of hyperparameters)
        permutation_builder = []
        for key, value in self.hyperparams.items():
            permutation_builder.append(range(len(value)))
        # Get all possible permutations from the permutations builder
        permutations = list(itertools.product(*permutation_builder))
        permutations = [list(a) for a in permutations]
        # Shuffle prevents it from being a grid search
        return random.shuffle(permutations)
