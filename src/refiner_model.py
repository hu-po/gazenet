import tensorflow as tf
import tensorflow.contrib.slim as slim

'''

The refiner network 'refines' a synthetic image, making it more real.

The input:
* A synthetic image

The output:
* The refined synthetic image

'''

class RefinerModel(object):

    def __init__(self, config):
        self.config=config

    def refiner_loss(self):
        pass

    def discrim_loss(self):
        pass
