import os
import time
import argparse
import tensorflow as tf
import multiprocessing as mp
import cv2

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.config import Config
from src.utils.cam_utils import WebcamVideoStream, FPS

'''
This file is used to run the Gaze Net. It takes images from your webcam, feeds them through the model
and outputs your current gaze location on the screen. Make sure to fullscreen it!

Sources:
[1] https://github.com/datitran/object_detector_app
'''


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    # Set up input and output queues for images
    input_q = mp.Queue(maxsize=args.queue_size)
    output_q = mp.Queue(maxsize=args.queue_size)
    pool = mp.Pool(args.num_workers, worker, (input_q, output_q))

    # Start up webcam stream and fps tracker
    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()
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
    # Clean up threads, camera streams, etc
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
