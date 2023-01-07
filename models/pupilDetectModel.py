from torch import nn
from torchvision.models.segmentation import fcn_resnet50
import torch

# TODO model do wykrywania środka źrenicy, na wyjsciu 2 pola regresji (wspolrzedne)
def pupilSegmentationModel():
    model = fcn_resnet50(weights=None, num_classes=1)
    return model
