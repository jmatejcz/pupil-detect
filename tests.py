from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import torch
from gaze_tracker import GazeTracker

###################################################################
####### GROUND TRUTH TEST #########################################
# Odległość od monitora - 94cm , Y = 0, X = 0 (wziac pod uwage rozstaw oczu) po 31mm na oko
# Wielkość monitora - 121.0, 68.4
# Znormalizowane pozycje znaczników w X i Y - 0.25, 0.5, 0.75

PATH = "pupil-detect/datasets/PupilCoreDataset/"
# PATH = "datasets/PupilCoreDataset/"

dataset = PupilCoreDatasetGazeTrack(
    f"{PATH}video5_eye0_video.avi",
    f"{PATH}video5_eye0_pupildata.csv",
    f"{PATH}video5_eye1_video.avi",
    f"{PATH}video5_eye1_pupildata.csv",
    dataset_len=1000,
)
WEIGHT_PATH = "pupil-detect/models/weights"
# WEIGHT_PATH = "models/weights"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_tracker = GazeTracker(weight_path=WEIGHT_PATH)
gaze_tracker.fit_tracker(dataset)
gaze_tracker.track_gaze_vector(dataset)
