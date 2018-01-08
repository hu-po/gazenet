import os
import sys
import tensorflow as tf

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(mod_path)

import src.utils.base_utils as base_utils

'''
Base model class is inherited to re-use some common code
'''


class BaseModel(object):

    @base_utils.config_checker(['image_width',
                                'image_height',
                                'image_channels',
                                'grayscale'])
    def __init__(self, config=None):
        self.config = config
        self.image = tf.placeholder(tf.float32, shape=(None,
                                                       config.image_height,
                                                       config.image_width,
                                                       1 if config.grayscale else config.image_channels),
                                    name='input_image')
