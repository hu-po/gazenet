import torchvision
import torch.nn as nn


def gazenet_model(**kwargs):
    """
    Creates a Gazenet model using pre-trained feature extractor and a new FC head
    :return: pytorch model
    """
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor

    # Modify the model head
    model.avgpool = nn.AdaptiveAvgPool2d(1)  # Allows for different input sizes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    if kwargs['use_gpu']:  # Convert to GPU-enabled
        model = model.cuda()

    return model
