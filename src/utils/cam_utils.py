import cv2
import datetime
import screeninfo
import numpy as np
from threading import Thread

'''
This file contains utils for reading images from a webcam using OpenCV, as well as for FPS calculation.

Sources:
 [1] http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
 [2] https://github.com/datitran/object_detector_app
'''


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self._stream = cv2.VideoCapture(src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self._stream.read()

        # indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self._stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def screen_plot(gaze_loc, image=None, radius=20):
    # Get monitor size using screeninfo pkg
    screen = screeninfo.get_monitors()[0]

    # Create blank canvas in which to plot gaze location
    canvas = np.zeros((screen.height, screen.width, 3), dtype=np.float32)

    # Plot image on center of canvas
    if image is not None:
        image = image / 255
        image_loc = [int(0.5 * screen.width - image.shape[0] / 2),
                     int(0.5 * screen.height - image.shape[1] / 2)]
        canvas[image_loc[0]:image_loc[0] + image.shape[0], image_loc[1]:image_loc[1] + image.shape[1]] = image

    # Plot gaze location
    x = int(gaze_loc[0] * screen.width)
    y = int(screen.height - gaze_loc[1] * screen.height)
    cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)

    # Plot information as text below
    text = 'Gaze location, %s x width   %s y height' % (gaze_loc[0], gaze_loc[1])
    text_loc = (int(0.8 * screen.height), int(0.5 * screen.width))
    cv2.putText(canvas, text, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    window_name = 'GazeTracker'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    return window_name, canvas
