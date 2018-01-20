import os
import sys
import tensorflow as tf

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

    def next_estimator(self):
        # Model params from hyperparameters
        run_name = self.model.next_run()
        model_params = self.model.build_model_params()
        # Model directory based on hyperparams
        model_dir = os.path.join(self.model_dir, run_name)
        self.make_path(model_dir)
        # Create new runconfig object
        run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                            save_summary_steps=self.save_summary_steps,
                                            save_checkpoints_steps=self.save_checkpoints_steps,
                                            keep_checkpoint_max=1)
        # Create a new estimator object
        estimator = tf.estimator.Estimator(model_fn=model_func_bank.resnet_gaze_model_fn,
                                           params=model_params,
                                           config=run_config)
        return estimator

    @staticmethod
    def make_path(path):
        if not os.path.exists(path):
            os.mkdir(path)
