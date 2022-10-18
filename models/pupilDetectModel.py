from torchvision import models
from torch import nn


def PupilDetectModel():
    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=3)
    return model
