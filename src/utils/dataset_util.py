import os

'''
This file file contains common functions used for directory and dataset manipulation.
'''

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# Local file directories
data_dir = os.path.join(root_dir, 'data')
log_dir = os.path.join(root_dir, 'logs')
model_dir = os.path.join(root_dir, 'models')