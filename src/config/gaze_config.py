import os
import sys

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import Config


class GazeCollectConfig(Config):
    # Device index of camera
    video_source = 0
    # Name of the dataset to collect
    dataset_name = '080118_real'


class GazeRunConfig(Config):
    # Device index of camera
    video_source = 0
    # Number of worker threads
    num_workers = 2
    # Size of the image queue
    queue_size = 5
