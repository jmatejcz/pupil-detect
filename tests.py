import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import torch
from eye_model import EyeModeling
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import utils
from models.ifOpened import ifOpenedModel
from models.trainers import PupilSegmentationTrainer, IfOpenedTrainer
from visualization import visualise_pupil
from gaze_tracker import GazeTracker

###################################################################
####### GROUND TRUTH TEST #########################################
# Odległość od monitora - 94cm , Y = 0, X = 0 (wziac pod uwage rozstaw oczu) po 31mm na oko
# Wielkość monitora - 121.0, 68.4
# Znormalizowane pozycje znaczników w X i Y - 0.25, 0.5, 0.75

PATH = "datasets/PupilCoreDataset/"

dataset = PupilCoreDatasetGazeTrack(
    f"{PATH}video5_eye0_video.avi",
    f"{PATH}video5_eye0_pupildata.csv",
    f"{PATH}video5_eye1_video.avi",
    f"{PATH}video5_eye1_pupildata.csv",
    dataset_len=1000,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_tracker = GazeTracker()
gaze_tracker.fit_tracker(dataset)
gaze_tracker.track_gaze_vector(dataset)
