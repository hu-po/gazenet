import os
import sys
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import cv2

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.config.gaze_run_config as CONF
from src.utils.cam_utils import WebcamVideoStream, FPS

'''
This file is used to run the Gaze Net. It takes images from your webcam, feeds them through the model
and outputs your current gaze location on the screen. Make sure to fullscreen it!

Sources:
[1] https://github.com/datitran/object_detector_app
'''


def gaze_inference(image_np, sess, model_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get input and output to the gaze model
    image_input = model_graph.get_tensor_by_name('input_image')
    predict = model_graph.get_tensor_by_name('predict')

    # Actual detection.
    gaze_output = sess.run(predict, feed_dict={image_input: image_np_expanded})

    # Visualization of the results of a detection.
    cam_utils.visualize_gaze_output()
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(CONF.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=model_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(gaze_inference(frame_rgb, sess, detection_graph))


if __name__ == '__main__':

    # Set up input and output queues for images
    input_q = mp.Queue(maxsize=CONF.queue_size)
    output_q = mp.Queue(maxsize=CONF.queue_size)
    pool = mp.Pool(CONF.num_workers, worker, (input_q, output_q))

    # Start up webcam stream and fps tracker
    video_capture = WebcamVideoStream(src=CONF.video_source, width=CONF.image_width, height=CONF.image_height).start()
    fps = FPS().start()

    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)
        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
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
