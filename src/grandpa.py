import os

'''
The Grandpa class is inherited to have some very common parameters
'''


class Grandpa(object):
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
