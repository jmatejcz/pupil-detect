from torchvision import models
from torch import nn
from torchvision.models.squeezenet import SqueezeNet1_1_Weights

# 1st model for checking if eye is opened
def ifOpenedModel():
    num_classes = 2
    model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
    classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        final_conv,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    model.classifier = classifier
    return model
