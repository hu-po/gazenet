import os
import sys
import time
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

from src.config.config import Config
import src.models.model_func_bank as model_func_bank
from src.dataset import Dataset
import src.utils.train_utils as train_utils

'''
This file is used to train the gaze net. It contains functions for reading
and decoding the data (should be in TFRecords format).
'''


def run_training(config=None):
    """
        Train gaze_trainer for the given number of steps.
    """
    model = GazeModel(config=config)

    with model.graph.as_default():
        # train and test iterators, need dataset to create feedable iterator
        train_iterator, train_batch_op = train_utils.input_feed(config=config.train_dataset)
        test_iterator, test_batch_op = train_utils.input_feed(config=config.test_dataset)
        # Summary and initializer ops
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        summary = tf.summary.merge(model.summaries)

    # Early stopping params
    best_loss = config.best_loss
    steps_since_loss_decrease = -1

    # with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
    with tf.Session(graph=model.graph) as sess:
        # Initialize variables
        sess.run(init_op)
        # Model saver and log writers
        test_writer = tf.summary.FileWriter(os.path.join(config.gaze_model.run_log_path, 'test'), sess.graph)
        train_writer = tf.summary.FileWriter(os.path.join(config.gaze_model.run_log_path, 'train'), sess.graph)
        for epoch_idx in range(config.num_epochs):
            # Training
            epoch_train_start = time.time()
            num_train_steps = 0
            sess.run(train_iterator.initializer)
            # TODO: Learning rate decay based on epoch
            try:  # Keep feeding batches in until OutOfRangeError (aka one epoch)
                while True:
                    image_batch, label_batch = sess.run(train_batch_op)
                    _, loss, train_summary = sess.run([model.optimize,
                                                       model.loss,
                                                       summary], feed_dict={model.image: image_batch,
                                                                            model.label: label_batch,
                                                                            model.is_training: True})
                    num_train_steps += 1
            except tf.errors.OutOfRangeError:
                epoch_train_duration = time.time() - epoch_train_start
                print('Epoch %d: Training (%.3f sec)(%d steps) - loss: %.2f' % (epoch_idx,
                                                                                epoch_train_duration,
                                                                                num_train_steps,
                                                                                loss))
            # if config.save_model and ((epoch_idx + 1) % config.save_every_n_epochs) == 0:
            #     save_path = saver.save(sess, os.path.join(config.checkpoint_path, str(epoch_idx + 1)))
            #     print('Model checkpoint saved at %s' % save_path)
            # Testing (one single batch is run)
            epoch_test_start = time.time()
            sess.run(test_iterator.initializer)
            image_batch, label_batch = sess.run(test_batch_op)
            loss, test_summary = sess.run([model.loss,
                                           summary], feed_dict={model.image: image_batch,
                                                                model.label: label_batch,
                                                                model.is_training: False})

            epoch_test_duration = time.time() - epoch_test_start
            print('Epoch %d: Testing (%.3f sec) - loss: %.2f' % (epoch_idx,
                                                                 epoch_test_duration,
                                                                 loss))

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
            # Write summary to log file
            test_writer.add_summary(test_summary, (epoch_idx + 1))
            train_writer.add_summary(train_summary, (epoch_idx + 1))
        # Close the log writers
        test_writer.close()
        train_writer.close()
    # Clean out the trash
    tf.reset_default_graph()


if __name__ == '__main__':
    # Create config and convert dataset to usable form
    config = Config.from_yaml('run/gaze_train.yaml')

    # Get the input feed tensors and initializer hooks for the datasets
    train_input_func, train_init_hook = Dataset.from_yaml('datasets/synthetic_gaze_train_small.yaml').input_feed()
    test_input_func, test_init_hook = Dataset.from_yaml('datasets/synthetic_gaze_test.yaml').input_feed()

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_func_bank.resnet_gaze_model_fn,
                                params=config.model_params,
                                model_dir=config.model_dir)

    # Logging hook
    tensors_to_log = {"input_image": "input_image",
                      "output": "output"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    # Train for one Epoch
    nn.train(input_fn=train_input_func,
             hooks=[train_init_hook, logging_hook])

    # Test
    ev = nn.evaluate(input_fn=test_input_func,
                     hooks=[test_init_hook, logging_hook])
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])
