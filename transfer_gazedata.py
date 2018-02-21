import argparse
import sys
from pathlib import Path

# Import local files and utils
sys.path.append(str(Path.cwd() / '..' / '..'))
import src.data_utils as data_utils
import src.saved_models as models
import src.train_utils as train_utils

parser = argparse.ArgumentParser(description='Gazenet Trainer')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# model hyper-params


if __name__ == '__main__':
    # Parse and print out parameters
    args = parser.parse_args()
    print('Running Gazenet Trainer. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    train_data, dev_data, embeddings = data_utils.load_dataset(args)

    train_utils.train_model(train_data, dev_data, model, args)