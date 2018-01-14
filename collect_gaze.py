import os
import sys
import time
import random
import cv2

mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mod_path)

from src.config.gaze_config import GazeCollectConfig
from src.utils.cam_utils import WebcamVideoStream

'''
This python file collects unlabeled real gaze images using the webcamera.

Sources:
[1] https://github.com/datitran/object_detector_app
'''

# Create config instance
CONF = GazeCollectConfig()

if __name__ == '__main__':


    # Create local data dir
    dataset_dir = os.path.join(CONF.data_dir, CONF.dataset_name)

    # Start up webcam stream
    video_capture = WebcamVideoStream(src=CONF.video_source, width=CONF.width, height=CONF.height).start()

    # Counter variable
    count = 1
    while True:

        # Pick a random gaze location
        gaze_x = random.uniform(0, 1)
        gaze_y = random.uniform(0, 1)

        # Plot gaze location onto screen

        # Plot countdown
        time.sleep(1)

        print('Taking image %s ' % count)
        frame = video_capture.read()

        # Plot recorded image and gaze location
        cv2.imshow('Video', frame)

        # Save image
        filename = '%.2f_%.2f.png' % (gaze_x, gaze_y)
        cv2.imwrite(os.path.join(dataset_dir, filename), frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Data collection over, took %d images into %s' % (count, dataset_dir))
    # Clean up threads, camera streams, etc
    video_capture.stop()
    cv2.destroyAllWindows()
