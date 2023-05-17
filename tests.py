from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import numpy as np
import utils
import json


################################################################################################
####### GROUND TRUTH TEST ######################################################################
# Tests are meant for videos of eyes staring and the 9 markers at a screen
# During recording head has to be in the same place
# Between switching gaze to next marker the longer blink (for example 1 second) is required

DATASET_PATH = "datasets/PupilCoreDataset/"
DATASET_LEN = 3450

WEIGHT_PATH = "models/weights"
RESULT_PATH = "results/ground_truth_test"

DIST_FROM_SCREEN = 940  # in mm
SCREEN_SIZE = (1210, 684)  # in mm
MARKERS_POSITIONS = [0.25, 0.5, 0.75]  # positions in the screen
PUPILARY_DISTANCE = 63  # in mm  average distance bettween pupils for adults


dataset_number = "4"
dataset = PupilCoreDatasetGazeTrack(
    f"{DATASET_PATH}{dataset_number}/eye0_video_rotated.avi",
    f"{DATASET_PATH}{dataset_number}/eye0_pupildata.csv",
    f"{DATASET_PATH}{dataset_number}/eye1_video.avi",
    f"{DATASET_PATH}{dataset_number}/eye1_pupildata.csv",
    DATASET_LEN,
)

############################################################################################
##################### COMPARING RESULTS WITH GROUND TRUTH ##################################
# to compare reslt with ground truth we need gaze vectors to be directed straight to screen
# but camera isnt directed straight to the eye, so we have to caluclate rotation matrix to fix this
# vector are calculated from camera perspective

# we have to arbitrally say when the test in video begins
# in my videos its around 8th seconds but it has to be adjusted to every test
# videos of both eyes can be unsynchronized, in my videos left eye is 100 frames
# faster than right on avergare, so we have to account for that
TEST_BEGIN = 505  # in frames

if __name__ == "__main__":
    # create vectors from eye to every marker on the screen
    markers_pos = []
    for i in MARKERS_POSITIONS:
        for j in reversed(MARKERS_POSITIONS):
            markers_pos.append([[j, i]])
    markers_pos = np.array(markers_pos)
    markers_vectors_right, markers_vectors_left = utils.create_vectors_to_markers(
        markers_pos, DIST_FROM_SCREEN, SCREEN_SIZE, PUPILARY_DISTANCE
    )

    results = utils.serialize_results_to_match_markers(
        f"{RESULT_PATH}{dataset_number}.json", TEST_BEGIN, DATASET_LEN, -100
    )
    (
        camera_vectors_right,
        camera_vectors_left,
        camera_vectors_right_means,
        camera_vectors_left_means,
    ) = utils.get_camera_vectors_and_mean_vectors_from_results(results)

    right_rot_matrixes = utils.get_rotation_matrixes_between_screen_and_camera(
        cam_vectors=camera_vectors_right_means, screen_vectors=markers_vectors_right
    )
    left_rot_matrixes = utils.get_rotation_matrixes_between_screen_and_camera(
        cam_vectors=camera_vectors_left_means, screen_vectors=markers_vectors_left
    )

    right_angular_errors = []
    left_angular_errors = []
    with open(f"results/angular_errors_{dataset_number}.json", "w", newline="") as f:
        for idx, (right_marker, left_marker) in enumerate(
            zip(camera_vectors_right, camera_vectors_left)
        ):
            angular_errors = []
            for u, i in enumerate(right_marker):

                r_vec = i
                r_vec = np.dot(right_rot_matrixes[idx], r_vec)
                r_vec = utils.calculate_point_of_sight(
                    r_vec, [-PUPILARY_DISTANCE, 0, 0], DIST_FROM_SCREEN
                )

                err = utils.get_angular_error(r_vec, markers_vectors_right[idx])
                angular_errors.append(err)

            marker = {idx: {"mean_right_eye_angular_error": np.mean(angular_errors)}}
            right_angular_errors.append(marker)

            angular_errors = []
            for i in left_marker:

                l_vec = i
                l_vec = np.dot(left_rot_matrixes[idx], l_vec)
                l_vec = utils.calculate_point_of_sight(
                    l_vec, [PUPILARY_DISTANCE, 0, 0], DIST_FROM_SCREEN
                )

                err = utils.get_angular_error(l_vec, markers_vectors_left[idx])
                angular_errors.append(err)

            marker = {idx: {"mean_left_eye_angular_error": np.mean(angular_errors)}}
            left_angular_errors.append(marker)

        json.dump(
            {"Right_eye": right_angular_errors, "Left_eye": left_angular_errors}, f
        )
