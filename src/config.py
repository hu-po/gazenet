import os

'''
This python file defines parameters used throughout the module.
Please change them here rather than within individual files.
'''

# =========== #
#   Dataset   #
# =========== #
dataset_name = '04012018_headlook'
# Image file names are regexed to get targets
dataset_regex = '(\d.\d+)_(\d.\d+).png'
train_test_split = 0.95
image_width = 128
image_height = 96
image_channels = 3
# Grayscale images are quicker, and depending on problem color is not important
grayscale = True
# Brightness Augmentation
random_brigtness = True
brightnes_max_delta = 0.1
# Contrast Augmentation
random_contrast = True
contrast_lower = 0.01
contrast_upper = 0.2

# ============ #
#   Training   #
# ============ #
num_epochs = 10
batch_size = 16
# Number of train examples (if you want to limit training data)
num_train_examples = 1000
# Number of test examples in each validation step
num_test_examples = 100
# Bigger buffer means better shuffling but slower start up and more memory used.
buffer_size = 300
learning_rate = 0.01
save_model = True


# ============== #
#    Directory   #
# ============== #
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Local file directories
data_dir = os.path.join(root_dir, 'data')
log_dir = os.path.join(root_dir, 'logs')
model_dir = os.path.join(root_dir, 'models')
# Dataset specific files
dataset_path = os.path.join(data_dir, dataset_name)
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')
train_tfrecord_path = os.path.join(dataset_path, 'train.tfrecords')
test_tfrecord_path = os.path.join(dataset_path, 'test.tfrecords')
log_path = os.path.join(log_dir, dataset_name)
checkpoint_path = os.path.join(model_dir, dataset_name)