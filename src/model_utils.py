from itertools import chain
import torch
import torchvision
import torch.nn as nn


class GazeNet(nn.Module):
    """
    Custom Gazenet model is made up of a feature extractor and a custom head.
    """

    def __init__(self, **kwargs):
        super(GazeNet, self).__init__()
        if kwargs['feature_extractor'] == 'resnet18':
            self.base = torchvision.models.resnet18(pretrained=True)
        elif kwargs['feature_extractor'] == 'resnet50':
            self.base = torchvision.models.resnet50(pretrained=True)
        else:
            raise Exception('Must provide a valid Feature extractor for GazeNet model')
        # Freeze the feature extractor
        if kwargs.get('freeze_base', False):
            for param in self.base.parameters():
                param.requires_grad = False

        # Modify the output of feature extractor to match head input
        self.base.avgpool = nn.AdaptiveAvgPool2d(1)  # Allows for different input sizes
        self.base.fc = nn.Linear(self.base.fc.in_features, kwargs['head_feat_in'])

        # Create model head
        self.head = nn.Linear(kwargs['head_feat_in'], 2)

        # Chain together parameter iterators for base and head
        self.optim_params = chain(self.base.fc.parameters(), self.head.parameters())

        if kwargs['use_gpu']:  # Convert to GPU-enabled
            self.base = self.base.cuda()
            self.head = self.head.cuda()

    def forward(self, x):
        """
        Feed forward function for our net
        :param x: (Variable) input data
        :return: (Variable) output data
        """
        features = self.base(x)
        output = self.head(features)
        return output