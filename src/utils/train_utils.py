import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self, init_op):
        super().__init__()
        self.init_op = init_op

    def after_create_session(self, session, coord):
        session.run(self.init_op)


class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self, init_op):
        super().__init__()
        self.init_op = init_op

    def after_create_session(self, session, coord):
        session.run(self.init_op)
