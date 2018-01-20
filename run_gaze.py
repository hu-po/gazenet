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
and outputs your current gaze location on the screen. Make sure to fullscreen it!

Sources:
[1] https://github.com/datitran/object_detector_app
'''

CHKPT = '/home/ook/repos/gazeGAN/local/models/gaze_resnet_rb_kernel_3_fc_layers_64_batch_norm_False_dimred_stride_4_rb_feat_8_num_rb_2_dimred_feat_32_dimred_kernel_4_learning_rate_0.01/checkpoint'

def gaze_inference(image_np, sess, model_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get input and output to the gaze model
    image_input = model_graph.get_tensor_by_name('input_image')
    predict = model_graph.get_tensor_by_name('predict')

    # Actual detection.
    gaze_output = sess.run(predict, feed_dict={image_input: image_np_expanded})

    # Visualization of the results of a detection.
    canvas = cam_utils.screen_plot(gaze_output, image=image_input, window_name=conf.window_name)
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
