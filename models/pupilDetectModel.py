from torchvision.models.segmentation import fcn_resnet50


def pupilSegmentationModel():
    # model used for segmentation
    model = fcn_resnet50(weights=None, num_classes=1)
    return model
