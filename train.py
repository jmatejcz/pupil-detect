import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDatasetTraining
import torch
from eye_model import EyeModeling
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
import numpy as np
import utils
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from models.trainers import PupilSegmentationTrainer, IfOpenedTrainer
from visualization.visualise_pupil import draw_normal_vectors_2D

DATASET_LEN_TO_USE = 5000
dataset = PupilCoreDatasetTraining(
    "datasets/PupilCoreDataset/video5_eye0_video.avi",
    "datasets/PupilCoreDataset/video5_eye0_pupildata.csv",
    "datasets/PupilCoreDataset/video5_eye1_video.avi",
    "datasets/PupilCoreDataset/video5_eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset.load_masks(
    "datasets/PupilCoreDataset/created_masks/eye0",
    "datasets/PupilCoreDataset/created_masks/eye1",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ifOpenedModel = ifOpenedModel()
pupilSegmentationModel = pupilSegmentationModel()

if_opened_trainer = IfOpenedTrainer(
    model=ifOpenedModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)
pupil_trainer = PupilSegmentationTrainer(
    model=pupilSegmentationModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)

pupil_trainer.train(device=device, num_epochs=5)
