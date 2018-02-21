import argparse
import sys
from pathlib import Path
import torch
from tensorboardX import SummaryWriter

# Import local files and utils
root_dir = Path.cwd()
sys.path.append(str(root_dir))
import src.data_utils as data_utils
import src.model_utils as model_utils
import src.train_utils as train_utils

parser = argparse.ArgumentParser(description='Gazenet Trainer')
# learning
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--lr_step_size', type=int, default=7,
                    help='step size in learning rate scheduler[default: 7]')
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='gamma in learning rate scheduler[default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum [default: 0.9]')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs for train [default: 25]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size for training [default: 8]')
# data
parser.add_argument('--datasets', type=str,
                    help='comma separated list of datasets to train on')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=128, help='Width of the images')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=96, help='Height of the images')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for data loader')
parser.add_argument('--saveas', type=str, default=None,
                    help='savename for trained model [default: None]')
# model params
parser.add_argument('--head_feat_in', type=int, default=256,
                    help='Number of input features for extra layer in head [default: 256]')
parser.add_argument('--feature_extractor', type=str, default='resnet18',
                    help='Feature extractor to use in model [default: resnet18]')

if __name__ == '__main__':
    # Parse and print out parameters
    args = parser.parse_args()
    print('Running Gazenet Trainer. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Make sure we can use GPU
    use_gpu = torch.cuda.is_available()
    print('Gpu is enabled: %s' % use_gpu)

    # Create model
    model = model_utils.GazeNet(use_gpu=use_gpu,
                                head_feat_in=args.head_feat_in,
                                feature_extractor=args.feature_extractor)

    # Create dataset
    dataloader = data_utils.gaze_dataloader(datasets=args.datasets,
                                            imsize=(args.height, args.width),
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    # Criterion (aka loss) is just simple MSE
    criterion = torch.nn.MSELoss()

    # We are only optimizing the head of the
    optimizer = torch.optim.SGD(model.optim_params, lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    # Write tensorboard logs
    if args.write_logs:
        writer = SummaryWriter(log_dir=)

    # Train the model
    trained_model = train_utils.train_gazenet(model, dataloader,
                                              criterion, optimizer, scheduler,
                                              num_epochs=args.num_epochs,
                                              use_gpu=use_gpu,
                                              writer=writer)

    # Save the model to local directory
    if args.saveas is not None:
        save_path = str(root_dir / 'src' / 'models' / args.saveas) + '.pt'
        print('Saving model to %s' % save_path)
        torch.save(trained_model, save_path)
        print('...done')

    print('Training Complete!')
