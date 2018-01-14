import os
import sys
import argparse
import time
import random
import cv2

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

import src.config.gaze_collect_config as CONF
from src.utils.cam_utils import WebcamVideoStream, FPS

'''
This python file collects unlabeled real gaze images using the webcamera.

Sources:
[1] https://github.com/datitran/object_detector_app
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    # Create local data dir
    dataset_dir = os.path.join(CONF.data_dir, CONF.dataset_name)

    # Start up webcam stream
    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()

    # Counter variable
    count = 1
    while True:
        frame = video_capture.read()

        print('Taking image %s ' % count)

        # Countdown on target gaze location
        time.sleep(1)

        # Save image
        filename = '%s.png' % i
        cv2.imwrite(os.path.join(dataset_dir, filename), frame)
        count += 1

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Data collection over, took %d images into %s' % (count, dataset_dir))
    # Clean up threads, camera streams, etc
    video_capture.stop()
    cv2.destroyAllWindows()
