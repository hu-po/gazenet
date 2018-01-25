import os
import sys
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config.config import Config
import src.utils.cam_utils as cam_utils
from src.utils.cam_utils import WebcamVideoStream, FPS

'''
This file is used to run the Gaze Net. It takes images from your webcam, feeds them through the model
and outputs your current gaze location on the screen.

Sources:
[1] https://github.com/datitran/object_detector_app
[2] https://github.com/tensorflow/serving/issues/488
'''

CHKPT = '/home/ook/repos/gazeGAN/local/models/a8cf9182-fb79-4a50-bdae-c7911f34598f/frozen_graph.pb'

'''

To freeze the graph use the following:

python freeze_graph.py \
--input_graph=graph.pbtxt \
--input_checkpoint=model.ckpt-0 \
--output_graph=frozen_graph.pb \
--output_node_names=gaze_resnet/output/BiasAdd \
--variable_names_blacklist=IsVariableInitialized,globalstep

'''

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

def gaze_inference(image_np, sess, model_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get input and output to the gaze model
    input_image = model_graph.get_tensor_by_name('gaze_resnet/input_image')
    predict = model_graph.get_tensor_by_name('gaze_resnet/output/BiasAdd')

    # Actual detection.
    gaze_output = sess.run(predict, feed_dict={input_image: image_np_expanded})

    # Visualization of the results of a detection.
    canvas = cam_utils.screen_plot(gaze_output, image=input_image, window_name=conf.window_name)
    return canvas


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    model_graph = tf.Graph()
    # model_checkpoint = os.path.join(conf.model_dir, conf.ckpt_dir, '/checkpoint')
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        # with tf.gfile.GFile(model_checkpoint, 'rb') as fid:
        with tf.gfile.GFile(CHKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=model_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        output_q.put(gaze_inference(frame, sess, model_graph))


if __name__ == '__main__':

    # Set up config object
    conf = Config('run/run_gaze.yaml')

    # Set up input and output queues for images
    input_q = mp.Queue(maxsize=conf.queue_size)
    output_q = mp.Queue(maxsize=conf.queue_size)
    pool = mp.Pool(conf.num_workers, worker, (input_q, output_q))

    # Start up webcam stream and fps tracker
    video_capture = WebcamVideoStream(src=conf.video_source, width=conf.image_width, height=conf.image_height).start()
    fps = FPS().start()

    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)
        output_frame = output_q.get()
        cv2.imshow(conf.window_name, output_frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print out fps tracker summary
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    # Clean up threads, camera streams, tf session, etc
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
    tf.reset_default_graph()
