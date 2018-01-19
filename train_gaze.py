import os
import sys
import time
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.config.config import Config
import src.models.model_func_bank as model_func_bank
from src.dataset import Dataset

'''
This file is used to train the gaze net. It contains functions for reading
and decoding the data (should be in TFRecords format).
'''


def run_training(config=None):
    """
        Train gaze_trainer according to given config.
    """
    assert config is not None, 'Please provide a config when running gaze trainer'
    config = Config.from_yaml(config)
    # Get hooks and feed functions for train and test datasets
    train_input_feed, train_init_hook = Dataset.from_yaml(config.train_dataset_yaml).feed_and_hook()
    test_input_feed, test_init_hook = Dataset.from_yaml(config.test_dataset_yaml).feed_and_hook()
    # Set up run config
    runconfig = tf.estimator.RunConfig(model_dir=config.model_dir,
                                       save_summary_steps=config.save_summary_steps,
                                       save_checkpoints_steps=config.save_checkpoints_steps,
                                       keep_checkpoint_max=1)

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_func_bank.resnet_gaze_model_fn,
                                params=config.model_params,
                                config=runconfig)
    # Early stopping params
    best_loss = config.best_loss
    steps_since_loss_decrease = -1
    # Loop train and testing for configured number of epochs
    for epoch_idx in range(config.num_epochs):
        # Training
        train_start = time.time()
        # Train for one Epoch
        nn.train(input_fn=train_input_feed, hooks=[train_init_hook])
        train_duration = time.time() - train_start
        # Testing (one single batch is run)
        test_start = time.time()
        ev = nn.evaluate(input_fn=test_input_feed, hooks=[test_init_hook])
        loss = ev['loss']
        rmse = ev['rmse']
        test_duration = time.time() - test_start
        print('Epoch %d: Training (%.3f sec) Testing (%.3f sec) - loss: %.2f - rmse: %.2f' % (epoch_idx,
                                                                                              train_duration,
                                                                                              test_duration,
                                                                                              loss,
                                                                                              rmse))
        # Early stopping breaks out if loss hasn't decreased in N steps
        if best_loss < loss:
            steps_since_loss_decrease += 1
        if loss < best_loss:
            best_loss = loss
            steps_since_loss_decrease = 0
        if steps_since_loss_decrease >= config.patience:
            print('Loss no longer decreasing, ending run')
            break
        # End early if the loss is way out of wack
        if loss > config.max_loss:
            print('Loss is too big off the bat, ending run')
            break


if __name__ == '__main__':
    run_training('run/gaze_train.yaml')
