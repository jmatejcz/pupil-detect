from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import torch
from gaze_tracker import GazeTracker
import numpy as np
import argparse


if __name__ == "__main__":
    DATASET_PATH = "datasets/PupilCoreDataset/"
    DATASET_LEN = 3450

    WEIGHT_PATH = "models/weights/"
    RESULT_PATH = "results/ground_truth_test"
    ##################################################################################
    ########### COLLECTING RESULTS ###################################################
    dataset_number = "4"
    dataset = PupilCoreDatasetGazeTrack(
        f"{DATASET_PATH}{dataset_number}/eye0_video_rotated.avi",
        f"{DATASET_PATH}{dataset_number}/eye0_pupildata.csv",
        f"{DATASET_PATH}{dataset_number}/eye1_video.avi",
        f"{DATASET_PATH}{dataset_number}/eye1_pupildata.csv",
        DATASET_LEN,
    )
    mm_to_px = np.linalg.norm([192, 192]) / np.linalg.norm([1.15, 1.15])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaze_tracker = GazeTracker(weight_path=WEIGHT_PATH, px_to_mm=mm_to_px)
    gaze_tracker.fit_tracker(dataset)
    gaze_tracker.track_gaze_vector(
        dataset, result_save_path=f"{RESULT_PATH}{dataset_number}.json"
    )
