import os
import datetime
import random
import itertools
from collections import OrderedDict

'''
This file contains utilities for config classes. Mostly to add more thangs.
'''


def config_checker(config_properties=None):
    """
    Decorator checks to make sure the function contains the neccessary config values
    :param config_properties: [string] list of strings of properties used in function
    :return: function
    """
    def decorator(func):
        def wrapped(*args, **kwargs):
            assert kwargs.get('config', None) is not None, '%s needs config argument' % func.__name__
            for prop in (config_properties or []):
                assert kwargs['config'].__getattribute__(prop) is not None, \
                    '%s needs the (not None) property %s' % (func.__name__, prop)
            return func(*args, **kwargs)
        return wrapped
    return decorator


def build_hyperparameter_config(config, exp_config_handle=None):
    # Runs are required when configs contain hyperparameters
    config.run_log_path = None
    config.run_checkpoint_path = None
    # List of all runs within experiment
    config.runs = []
    if exp_config_handle is not None:
        # Take log and checkpoint paths from experiment
        config.log_path = exp_config_handle.log_path
        config.checkpoint_path = exp_config_handle.checkpoint_path


def prepare_run(config, idx):
    config.set_hyperparams(idx)
    config.create_run_directories()


def generate_runs(config):
    # Generate all runs (all possible permutations of hyperparameters)
    permutation_builder = []
    for key, value in config.hyperparams.items():
        permutation_builder.append(range(len(value)))
    # Get all possible permutations from the permutations builder
    permutations = list(itertools.product(*permutation_builder))
    permutations = [list(a) for a in permutations]
    # Shuffle prevents it from being a grid search
    random.shuffle(permutations)
    # Add run-related properties to config class
    config.runs = permutations
    config.num_runs = len(config.runs)


def set_hyperparams(config, idx):
    permutation = config.runs[idx]
    config.run_hyperparams = OrderedDict()
    for i, key in enumerate(config.hyperparams.keys()):
        value = config.hyperparams[key][permutation[i]]
        config.run_hyperparams[key] = value
        setattr(config, key, value)


def create_run_directories(config):
    run_specific_name = config.model_name
    for key, value in config.run_hyperparams.items():
        str_value = str(value)
        if isinstance(value, list):
            str_value = '_'.join(str(a) for a in value)
        run_specific_name += '_%s_%s' % (key, str_value)
    config.run_log_path = os.path.join(config.log_path, run_specific_name)
    make_path(config.run_log_path)
    config.run_checkpoint_path = os.path.join(config.checkpoint_path, run_specific_name)
    make_path(config.run_checkpoint_path)


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def build_dataset_config(config):


def build_experiment_config(config):
    d = datetime.datetime.today()
    experiment_name = '%s_%sm_%sd_%shr_%smin' % (config.experiment_name, d.month, d.day, d.hour, d.minute)
    # Create experiment specific log and checkpoint directories
    config.log_path = os.path.join(config.log_dir, experiment_name)
    config.checkpoint_path = os.path.join(config.model_dir, experiment_name)
    make_path(config.log_path)
    make_path(config.checkpoint_path)
    print('Created log and checkpoint directories for experiment %s' % experiment_name)
