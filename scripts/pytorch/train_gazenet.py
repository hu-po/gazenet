import argparse
import sys
from pathlib import Path
import torch

# Import local files and utils
root_dir = Path.cwd()
sys.path.append(str(root_dir))
import pytorch.data_utils as data_utils
import pytorch.model_utils as model_utils
import pytorch.train_utils as train_utils

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
parser.add_argument('--dataset', type=str,
                    help='gaze dataset to train on')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for data loader')
parser.add_argument('--saveas', type=str, default=None,
                    help='savename for trained model [default: None]')


if __name__ == '__main__':
    # Parse and print out parameters
    args = parser.parse_args()
    print('Running Gazenet Trainer. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Make sure we can use GPU
    use_gpu = torch.cuda.is_available()
    print('Gpu is enabled: %s' % use_gpu)

    # Grab model and dataloader from utilities
    model = model_utils.gazenet_model(use_gpu=use_gpu)
    dataloader = data_utils.gaze_dataloader(dataset_name=args.dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    # Criterion (aka loss) is just simple MSE
    criterion = torch.nn.MSELoss()

    # We are only optimizing the head of the
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    # Train the model
    trained_model = train_utils.train_gazenet(model, dataloader,
                                              criterion, optimizer, scheduler,
                                              num_epochs=args.num_epochs,
                                              use_gpu=use_gpu)

    # Save the model to local directory
    if args.saveas is not None:
        save_path = str(root_dir / 'pytorch' / 'saved_models' / args.saveas)
        print('Saving model to %s' % save_path)
        torch.save(trained_model, save_path)
        print('...done')

    print('Training Complete!')