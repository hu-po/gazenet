import os
import sys
import time

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.trainer import Trainer

'''
This file is used to train the gaze net.
'''


def run_training(yaml_name=None):
    """
        Train gaze_trainer according to given config.
    """
    assert yaml_name is not None, 'Please provide a trainer yaml when running gaze trainer'
    trainer = Trainer(yaml_name)
    # Iterate through all the possible runs in the trainer
    for _ in trainer.model.runs:
        # Get the next estimator object from the trainer object
        estimator = trainer.next_estimator()
        try:
            # Early stopping params
            best_loss = trainer.best_loss
            steps_since_loss_decrease = -1
            # Loop train and testing for configured number of epochs
            for epoch_idx in range(trainer.num_epochs):
                # Training
                train_start = time.time()
                # Train
                estimator.train(input_fn=trainer.train_dataset.input_feed)
                train_duration = time.time() - train_start
                # Testing
                test_start = time.time()
                ev = estimator.evaluate(input_fn=trainer.test_dataset.input_feed)
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
                if steps_since_loss_decrease >= trainer.patience:
                    print('Loss no longer decreasing, ending run')
                    break
                # End early if the loss is way out of wack
                if loss > trainer.max_loss:
                    print('Loss is too big off the bat, ending run')
                    break
            # Add run stats to the train history dataframe
            trainer.update_history(epoch_idx, loss, rmse)
        except Exception:
            print('Something went wrong, trying the next config')

if __name__ == '__main__':
    run_training('run/gaze_train_synth.yaml')
