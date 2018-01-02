import cv2
import random
import time
import os
import sys

mod_path = os.path.abspath(os.path.join('..'))
sys.path.append(mod_path)
from src.config import DATA_DIR

'''
This python file collects gaze data using the webcamera.
'''

# Dataset parameters
DATASET_NAME = '020118_fingers'
NUM_IMAGES = 50

# Create local data dir
dataset_dir = os.path.join(DATA_DIR, DATASET_NAME)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# Possible gaze_locations
gaze_locations = [1, 2, 3, 4]

for i in range(NUM_IMAGES):
    # Plot target gaze location on screen
    label = random.choice(gaze_locations)
    print('Upcoming gaze location: ', label)

    # Set up video capture device on webcam
    webcam = cv2.VideoCapture(0)

    # Countdown on target gaze location
    time.sleep(3)

    # Snap image from webcamera
    snap = webcam.grab()
    ret, frame = webcam.retrieve()

    # # Show image
    # cv2.imshow('Video', frame)
    # cv2.waitKey()

    # Save image
    filename = '%s_%s.jpg' % (i, label)
    cv2.imwrite(os.path.join(dataset_dir, filename), frame)

    # Delete camera object to reset buffer
    del webcam
