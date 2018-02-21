import argparse
import sys
from pathlib import Path
import cv2
import torch

# Import local files and utils
root_dir = Path.cwd()
sys.path.append(str(root_dir))
import scripts.cam_utils as cam_utils
import pytorch.data_utils as data_utils

'''
This file is used to run the Gaze Net. It takes images from your webcam, feeds them through the model
and outputs your current gaze location on the screen.
'''

parser = argparse.ArgumentParser(description='Gazenet Runner')
parser.add_argument('--model', type=str, default=None,
                    help='Model to run[default: None]')
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=128, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=96, help='Height of the frames in the video stream.')
parser.add_argument('--window_name', type=str, default='GazeNet',
                    help='Name of window for when running [default: GazeNet]')
args = parser.parse_args()


def gaze_inference(image_np, model):
    # Convert input image from numpy
    input_image = data_utils.ndimage_to_variable(image_np,
                                                 imsize=(args.height, args.width),
                                                 use_gpu=True)
    # Inference (how about them type conversions)
    gaze_output = model(input_image).cpu().data.numpy().tolist()[0]
    # Visualization of the results of a detection.
    canvas = cam_utils.screen_plot(gaze_output, image=image_np, window_name=args.window_name)
    return canvas


if __name__ == '__main__':
    # Print out parameters
    print('Gazenet Model Runner. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Load Pytorch model from saved models directory
    model_path = str(Path.cwd() / 'pytorch' / 'saved_models' / args.model)
    print('Loading model from %s' % model_path)
    model = torch.load(model_path)

    # Start up webcam stream and fps tracker
    video_capture = cam_utils.WebcamVideoStream(src=args.video_source,
                                                width=args.width,
                                                height=args.height).start()
    fps = cam_utils.FPS().start()
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        output_frame = gaze_inference(frame, model)
        cv2.imshow(args.window_name, output_frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print out fps tracker summary
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    # Clean up camera streams, cv2 windows, etc
    video_capture.stop()
    cv2.destroyAllWindows()
