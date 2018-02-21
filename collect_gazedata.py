import sys
import argparse
from pathlib import Path
import datetime
import random
import cv2

# Import local files and utils
root_dir = Path.cwd()
sys.path.append(str(root_dir))
from src.cam_utils import WebcamVideoStream, screen_plot

'''
This python file collects real gaze images using the webcamera.
'''

parser = argparse.ArgumentParser(description='Gaze Dataset Collector')
parser.add_argument('--dataset', type=str, default=datetime.date.today().strftime('gaze_real_%y%m%d'),
                    help='Name to give to dataset[default: None]')
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=128, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=96, help='Height of the frames in the video stream.')
parser.add_argument('--window_name', type=str, default='Gaze Dataset Collector',
                    help='Name of window for when running [default: Gaze Dataset Collector]')
args = parser.parse_args()

if __name__ == '__main__':
    # Print out parameters
    print('Gaze Dataset Collector Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Create dataset directory
    dataset_dir = root_dir / 'data' / args.dataset / 'train'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Start up webcam stream
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    # Counter variable
    count = 1
    while True:
        # Pick a random gaze location
        gaze_x = random.uniform(0, 1)
        gaze_y = random.uniform(0, 1)

        # Plot gaze location onto screen
        canvas = screen_plot([gaze_x, gaze_y],
                             extra_text='LOOK AT TARGET \n AND PRESS ANY KEY',
                             window_name=args.window_name)
        cv2.imshow(args.window_name, canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()

        frame = video_capture.read()

        # Plot recorded image and gaze location
        canvas = screen_plot([gaze_x, gaze_y],
                             image=frame,
                             extra_text='Image %s taken \n (q) QUIT \n (s) SKIP \n (*) CONTINUE',
                             window_name=args.window_name)
        cv2.imshow(args.window_name, canvas)
        key = cv2.waitKey() & 0xFF
        cv2.destroyAllWindows()

        if key == ord('s'):
            print('Image skipped')
            continue
        else:
            print('Saved image %d' % count)
            filename = '%.2f_%.2f.png' % (gaze_x, gaze_y)
            cv2.imwrite(str(dataset_dir / filename), frame)
            count += 1

        if key == ord('q'):
            print('Quit gaze collection')
            break

    print('Data collection over, took %d images into %s' % (count, str(dataset_dir)))
    # Clean up threads, camera streams, etc
    video_capture.stop()
    cv2.destroyAllWindows()
