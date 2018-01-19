import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self, iterator=None):
        super().__init__()
        self.iterator = iterator

    def after_create_session(self, session, coord):
        session.run(self.iterator.initializer)

    # def before_run(self, run_context):
    #     return self.iterator.get_next()


class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self):
        super().__init__()

    def after_run(self, run_context, run_values):
        if True:
            run_context.request_stop()
