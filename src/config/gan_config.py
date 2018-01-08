import os

'''
This python file defines parameters used to train the GAN (refiner and discriminator nets)
'''

# =========== #
#   Dataset   #
# =========== #
synthetic_dataset = '04012018_headlook'
real_dataset = '070118_real'
# Clip the dataset sizes
num_synth_images = 5000
num_real_images = 100
# Image dimensions
image_width = 128
image_height = 96
image_channels = 3
# Grayscale images are quicker, and depending on problem color is not important
grayscale = True

# ============ #
#   Training   #
# ============ #
num_training_steps = 400
num_refiner_steps = 50  # Kg
num_discriminator_steps = 1  # Kd

synth_batch_size = 16
real_batch_size = 16

# Bigger buffer means better shuffling but slower start up and more memory used.
buffer_size = 100
learning_rate = 0.01

# Save model checkpoint
save_model = True
save_every_n_epochs = 50
# Model dropout
dropout_keep_prob = 0.8

# ============== #
#    Directory   #
# ============== #
dataset_path = os.path.join(data_dir, dataset_name)
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')
train_tfrecord_path = os.path.join(dataset_path, 'train.tfrecords')
test_tfrecord_path = os.path.join(dataset_path, 'test.tfrecords')
log_path = os.path.join(log_dir, dataset_name)
checkpoint_path = os.path.join(model_dir, dataset_name)
