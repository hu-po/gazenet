import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self, iterator=None):
        super().__init__()
        self.iterator = iterator

    def after_create_session(self, session, coord):
        session.run(self.iterator.initializer)
