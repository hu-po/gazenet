import os
import sys
import time
import random
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config.config import Config
from src.utils.cam_utils import WebcamVideoStream, screen_plot

'''
This python file collects unlabeled real gaze images using the webcamera.

Sources:
[1] https://github.com/datitran/object_detector_app
'''

if __name__ == '__main__':
    # Set up config object
    conf = Config.from_yaml('collect_gaze.yaml')
    # Create local data dir
    dataset_dir = os.path.join(conf.data_dir, conf.dataset_name)
    # Start up webcam stream
    video_capture = WebcamVideoStream(src=conf.video_source, width=conf.image_width, height=conf.image_height).start()
    # Counter variable
    count = 1
    while True:
        # Pick a random gaze location
        gaze_x = random.uniform(0, 1)
        gaze_y = random.uniform(0, 1)

        # Plot gaze location onto screen
        canvas = screen_plot([gaze_x, gaze_y],
                             extra_text='LOOK AT TARGET \n AND PRESS ANY KEY',
                             window_name=conf.window_name)
        cv2.imshow(conf.window_name, canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()

        frame = video_capture.read()

        # Plot recorded image and gaze location
        canvas = screen_plot([gaze_x, gaze_y],
                             image=frame,
                             extra_text='Image %s taken \n (q) QUIT \n (s) SKIP \n (*) CONTINUE',
                             window_name=conf.window_name)
        cv2.imshow(conf.window_name, canvas)
        key = cv2.waitKey() & 0xFF
        cv2.destroyAllWindows()

        if key == ord('s'):
            print('Image skipped')
            continue
        else:
            print('Saved image %d' % count)
            filename = '%.2f_%.2f.png' % (gaze_x, gaze_y)
            cv2.imwrite(os.path.join(dataset_dir, filename), frame)
            count += 1

        if key == ord('q'):
            print('Quit gaze collection')
            break

    print('Data collection over, took %d images into %s' % (count, dataset_dir))
    # Clean up threads, camera streams, etc
    video_capture.stop()
    cv2.destroyAllWindows()
