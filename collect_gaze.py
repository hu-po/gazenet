import os
import sys
import time
import random
import cv2


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.cam_utils import WebcamVideoStream, screen_plot


'''
This python file collects unlabeled real gaze images using the webcamera.

Sources:
[1] https://github.com/datitran/object_detector_app
'''

if __name__ == '__main__':

    # Create local data dir
    # dataset_dir = os.path.join(CONF.data_dir, CONF.dataset_name)

    # Start up webcam stream
    # video_capture = WebcamVideoStream(src=CONF.video_source, width=CONF.width, height=CONF.height).start()
    video_capture = WebcamVideoStream(src=0, width=128, height=96).start()

    # Counter variable
    count = 1
    while True:

        # Pick a random gaze location
        gaze_x = random.uniform(0, 1)
        gaze_y = random.uniform(0, 1)

        # # Plot gaze location onto screen
        # window_name, canvas = screen_plot([gaze_x, gaze_y])
        # cv2.imshow(window_name, canvas)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # Plot countdown
        time.sleep(1)

        print('Taking image %s ' % count)
        frame = video_capture.read()

        # Plot recorded image and gaze location
        window_name, canvas = screen_plot([gaze_x, gaze_y], image=frame)
        cv2.imshow(window_name, canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # # Save image
        # filename = '%.2f_%.2f.png' % (gaze_x, gaze_y)
        # cv2.imwrite(os.path.join(dataset_dir, filename), frame)
        # count += 1
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print('Data collection over, took %d images into %s' % (count, dataset_dir))
    # Clean up threads, camera streams, etc
    video_capture.stop()
    cv2.destroyAllWindows()
