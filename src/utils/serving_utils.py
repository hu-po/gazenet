import os
import tensorflow as tf

'''
This file contains utils for saving, loading, and serving an estimator model

Sources:
 [1] https://github.com/tensorflow/serving/issues/488
'''


def serving_input_receiver_fn():
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def save_tf_learn_model(estimator, model_name, export_dir):
    serving_input_fn = serving_input_receiver_fn
    export_dir = os.path.join(export_dir, model_name)
    estimator.export_savedmodel(export_dir, serving_input_fn)
    print("Done exporting tf.learn model to " + export_dir + "!")


SESS_DICT = {}


def get_session(model_id):
    global SESS_DICT
    config = tf.ConfigProto(allow_soft_placement=True)
    SESS_DICT[model_id] = tf.Session(config=config)
    return SESS_DICT[model_id]


def load_tf_model(model_path):
    sess = get_session(model_path)
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    return sess
