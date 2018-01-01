import cv2
import os
import sys
mod_path = os.path.abspath(os.path.join('..'))
sys.path.append(mod_path)
from src.config import DATA_DIR


'''
This python file collects gaze data using the webcamera.
'''

# Dataset parameters
DATASET_NAME = 'test'
NUM_IMAGES = 10

# Create local data dir
dataset_dir = os.path.join(DATA_DIR, DATASET_NAME)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# Use cv2 to save and
for _ in range(NUM_IMAGES):

    # Plot target gaze location on screen

    # Countdown on target gaze location

    # Snap image from webcamera
    video_capture = cv2.VideoCapture(0)

    # Extract frame
    ret, frame = video_capture.read()

    # Show image
    cv2.imshow('Video', frame)

    cv2.waitKey()
