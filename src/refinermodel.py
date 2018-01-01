import tensorflow as tf
from .helper_func import define_scope

'''

This network will be used to convert synthetic images into real images.
These will then be used to train the gaze network on the gaze task.

The input:
* A series of synthetic images `img_synth`
* A series of real images `img_real`

The output:
* A trained refiner network

'''

class RefinerModel(object):

    def __init__(self, img_synth, img_real):
        self.img_synth = img_synth
        self.img_real = img_real

    @define_scope
    def refiner_loss(self):
        pass

    @define_scope
    def discrim_loss(self):
        pass
