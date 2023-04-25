from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import torch
from gaze_tracker import GazeTracker

###################################################################
####### GROUND TRUTH TEST #########################################
# Odległość od monitora - 94cm , Y = 0, X = 0 (wziac pod uwage rozstaw oczu) po 31mm na oko
# Wielkość monitora - 121.0, 68.4
# Znormalizowane pozycje znaczników w X i Y - 0.25, 0.5, 0.75
DATASET_PATH = "datasets/PupilCoreDataset/"
DATASET_LEN = 3600
# dataset = PupilCoreDatasetGazeTrack(
#     f"{DATASET_PATH}video5_eye0_video.avi",
#     f"{DATASET_PATH}video5_eye0_pupildata.csv",
#     f"{DATASET_PATH}video5_eye1_video.avi",
#     f"{DATASET_PATH}video5_eye1_pupildata.csv",
#     DATASET_LEN,
# )
dataset_1 = PupilCoreDatasetGazeTrack(
    f"{DATASET_PATH}1/eye0_video_rotated.avi",
    f"{DATASET_PATH}1/eye0_pupildata.csv",
    f"{DATASET_PATH}1/eye1_video.avi",
    f"{DATASET_PATH}1/eye1_pupildata.csv",
    DATASET_LEN,
)
WEIGHT_PATH = "models/weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_tracker = GazeTracker(weight_path=WEIGHT_PATH)
gaze_tracker.fit_tracker(dataset_1)
# gaze_tracker.track_gaze_vector(
#     dataset_1, result_save_path="results/ground_truth_test1.csv"
# )
