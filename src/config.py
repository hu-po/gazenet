import os

'''
This python file defines parameters used throughout the module.
Please change them here rather than within individual files.
'''

# ============== #
#    Directory   #
# ============== #
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, 'data')
log_dir = os.path.join(root_dir, 'logs')
model_dir = os.path.join(root_dir, 'models')
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')
train_tfrecord =
test_tfrecord =

# =========== #
#   Dataset   #
# =========== #
dataset_name = '04012018_headlook'
# Image file names are regexed to get targets
dataset_regex = '(\d.\d+)_(\d.\d+).png'
train_test_split = 0.9
image_width = 128
image_height = 96
image_channels = 3
# Grayscale images are quicker, and depending on problem color is not important
grayscale = True

# ============ #
#   Training   #
# ============ #
num_epochs = 10
batch_size = 3
# Bigger buffer means better shuffling but slower start up and more memory used.
buffer_size = 10
learning_rate = 0.01

