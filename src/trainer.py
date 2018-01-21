import os
import sys
import datetime
from uuid import uuid4
import tensorflow as tf
import pandas as pd

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.dataset import Dataset
from src.models.model import Model
from src.config.config import Config
import src.models.model_func_bank as model_func_bank


class Trainer(Config):

    def __init__(self, yaml_name):
        super().__init__(yaml_name)
        # Create test and train dataset objects
        self.train_dataset = Dataset.from_yaml(self.train_dataset_yaml)
        self.test_dataset = Dataset.from_yaml(self.test_dataset_yaml)
        # Create model object
        self.model = Model(self.model_yaml)
        # Every hyperparameter run has an estimator
        self.estimator = None
        # Unique identifier string for each hyperparameter run
        self.run_id = None
        # Keep track of run performance and hyperparameters in a pandas dataframe
        history_columns = self.model.model_params + ['id', 'loss', 'steps', 'rmse']
        self.history = pd.DataFrame(columns=history_columns)
        # Save the history dataframe
        d = datetime.datetime.today()
        save_name = '%s_%sm_%sd_%shr_%smin' % (self.experiment_name, d.month, d.day, d.hour, d.minute)
        self.history_save_path = os.path.join(self.model_dir, save_name)

    def update_history(self, steps, loss, rmse):
        # Add loss and steps to model params dictionary
        model_params = self.model.model_param_dict()
        model_params['loss'] = loss
        model_params['steps'] = steps
        model_params['rmse'] = rmse
        model_params['id'] = self.run_id
        # Add the run to the history dataframe
        self.history = self.history.append(model_params, ignore_index=True)
        # Save pandas dataframe to file
        self.history.to_pickle(self.history_save_path + '.pdpckle')

    def next_estimator(self):
        # Model params from hyperparameters
        model_params = self.model.next_run()
        # Model directory is a randomly generated unique identifier string
        self.run_id = str(uuid4())
        model_dir = os.path.join(self.model_dir, self.run_id)
        self.make_path(model_dir)
        # Create new runconfig object
        run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                            save_summary_steps=self.save_summary_steps,
                                            save_checkpoints_steps=self.save_checkpoints_steps,
                                            keep_checkpoint_max=1)
        # Create a new estimator object
        self.estimator = tf.estimator.Estimator(model_fn=model_func_bank.resnet_gaze_model_fn,
                                                params=model_params,
                                                config=run_config)
        return self.estimator

    def save_estimator(self):
        export_dir = os.path.join(self.model_dir, self.run_id, 'exported_model')
        self.estimator.export_savedmodel(export_dir, self.serving_input_receiver_fn)
        print("Exported estimator model to " + export_dir)

    def serving_input_receiver_fn(self):
        feature_spec = {'input_image': tf.FixedLenFeature(dtype=tf.float32,
                                                          shape=[self.image_width,
                                                                 self.image_height, 1])}
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()

    @staticmethod
    def make_path(path):
        if not os.path.exists(path):
            os.mkdir(path)
