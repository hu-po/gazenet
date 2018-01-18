import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self, init_op=None):
        super().__init__()
        self.init_op = init_op

    def after_create_session(self, session, coord):
        session.run(self.init_op)


class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self):
        super().__init__()

    def after_run(self, run_context, run_values):
        if True:
            run_context.request_stop()
