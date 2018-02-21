from pathlib import Path
import re
from collections import namedtuple
import random
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_single_image(image_path, imsize):
    """
    Loads a single image from file
    :param image_path: (string) filepath for image
    :param imsize: (tuple) desired image size
    :return: (Variable) image tensor
    """
    # Scales to image size and converts to tensor
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    img = Image.open(image_path)
    # Strip out 4th channel
    img_array = np.array(img)
    img_stripped = img_array[:, :, :3]
    img = Image.fromarray(img_stripped)
    img = Variable(loader(img))
    # Fake a batch dimension
    img = img.unsqueeze(0)
    return img


class GazeDataset(Dataset):
    """Gaze Dataset Class. Inherits from PyTorch's Dataset class."""

    # Local dataset directory
    root_dir = Path.cwd()
    data_dir = root_dir / 'data'

    def __init__(self, dataset_name, transform=None, filename_regex='(\d.\d+)_(\d.\d+).png'):
        """
        :dataset_name: (string) name of directory containing all the images
        :filename_regex: (string) regex used to extract gaze data from filename
        :transform: (optional callable) transform to be applied to image
        """
        self.name = dataset_name
        self.data_path = GazeDataset.data_dir / dataset_name
        self.filename_regex = filename_regex
        self.transform = transform
        self.dataset = self._load_dataset()

    def _extract_target_from_gazefilename(self, imagepath):
        """
        Extract the label from the image path name
        :imagepath: (Path) image path (contains target)
        :return: tuple(int, int) gaze target
        """
        m = re.search(self.filename_regex, imagepath.name)
        gaze_x = float(m.group(1))
        gaze_y = float(m.group(2))
        return gaze_x, gaze_y

    def _load_dataset(self):
        """
        Loads dataset into a pandas dataframe
        :return: (pd) dataset with (filename, gaze_x, gaze_y) as header columns
        """
        # Use pathlib glob to get images
        image_list = list(self.data_path.glob('*.png'))
        print('Loading dataset %s, there are %s images.' % (self.data_path.name, len(image_list)))
        # Create new pandas dataframe
        df = pd.DataFrame(index=list(range(len(image_list))),
                          columns=['imagepath', 'gaze_x', 'gaze_y'])
        # Add all images in dataset folder into dataframe
        for i, imagepath in enumerate(image_list):
            gaze_x, gaze_y = self._extract_target_from_gazefilename(imagepath)
            df.loc[i] = [imagepath, gaze_x, gaze_y]
        return df

    def _get_datapoint(self, idx):
        """
        Returns a single datapoint at a given index
        :param idx: (int) index of datapoint to retreive
        """
        # Load image using PIL
        image = Image.open(self.dataset.iloc[idx, 0])
        # Strip out 4th channel
        img_array = np.array(image)
        img_stripped = img_array[:, :, :3]
        image = Image.fromarray(img_stripped)
        gaze_x = self.dataset.iloc[idx, 1]
        gaze_y = self.dataset.iloc[idx, 2]
        return image, gaze_x, gaze_y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets a single item of dataset based on index
        :param idx: (int) index of datapoint to retreive
        :return sample: (dict) image, gaze target
        """
        image, gaze_x, gaze_y = self._get_datapoint(idx)
        # Apply transform if necessary
        if self.transform:
            image = self.transform(image)
        # Put info into a dictionary
        sample = {'image': image, 'gaze_x': gaze_x, 'gaze_y': gaze_y}
        return sample

    def plot_samples(self, num_images=3):
        """
        Plots random sample images
        :num_images: (int) number of images to plot per dataset
        """
        fig = plt.figure()
        for i in range(num_images):
            sample_idx = random.randint(0, self.__len__())
            image, gaze_x, gaze_y = self._get_datapoint(sample_idx)
            # Use sublots to plot all of them
            ax = plt.subplot(1, num_images, i + 1)
            plt.tight_layout()
            plt.imshow(image)
            ax.set_title('Image %s: (%.2f, %.2f)' % (sample_idx, gaze_x, gaze_y))
            ax.axis('off')
        plt.show()


# Custom named tuple is used for retreiving DataLoader objects
GazeDataLoader = namedtuple('GazeDataLoader', ['dataloader', 'dataset'])


def gaze_dataloader(**kwargs):
    """
    Creates dataloaders to be used for training a gaze model
    :return: (dict of namedtuples) contains DataLoaders, GazeDataset
    """

    # Data transform define transformation applied to each image before being fed into model
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Populate the return dictionary with datasets and dataloaders
    return_dict = {}
    for phase in data_transforms.keys():
        dataset_name = str(Path(kwargs['dataset_name']) / phase)
        # Create dataset and dataloader objects
        dataset = GazeDataset(dataset_name, data_transforms[phase])
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=kwargs['batch_size'],
                                                 shuffle=True,
                                                 num_workers=kwargs['num_workers'])
        # Push into our custom namedtuple
        return_dict[phase] = GazeDataLoader(dataloader, dataset)
    return return_dict
