import cv2
import numpy as np
from exceptions import NoEllipseFound, NoIntersection
import math
import json


def fit_ellipse(mask):
    _, thresh = cv2.threshold(mask, 0.7, 1, cv2.THRESH_BINARY)
    thresh = np.asarray(thresh, dtype=np.uint8)

    countours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        max_area = 0
        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        ellipse = cv2.fitEllipse(max_cnt)
        # catch all weird invalid shapes
        if (
            ellipse[2] == 90.0
            or ellipse[2] == 0.0
            or ellipse[1][0] == 1.0
            or ellipse[1][0] == 0.0
        ):
            raise NoEllipseFound("Invalid ellipse shape")
        else:
            return ellipse
    except:
        raise NoEllipseFound("No ellipse found in mask")


def draw_ellipse(image, ellipse):
    cv2.ellipse(image, ellipse, (0, 0, 255), 1)
    return image


def calc_intersection(vectors, centers):
    """
    Calculate point nearest to all the lines, starting in centers
    """
    R_sum = 0
    S_sum = 0
    # normalize vectors
    vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors[0].shape[0]
    I = np.eye(dim)

    for i, vec in enumerate(vectors):
        R = I - np.matmul(vec, vec.transpose())
        R_sum += R

        S = np.matmul(R, centers[i].reshape(dim, 1))
        S_sum += S

    intersection = np.matmul(np.linalg.inv(R_sum), S_sum)

    return intersection


def calc_sphere_line_intersection(u, o, c, r):
    """Find intersection of line and sphere

    :param u: unit directio vector of line
    :param o: origin of line
    :param c: center of sphere
    :param r: radius of sphere
    """
    delta = np.square(np.dot(u.T, (o - c))) - np.dot((o - c).T, (o - c)) + np.square(r)
    if delta < 0:
        raise NoIntersection("No intersection between line and eye sphere")
    else:
        d = -np.dot(u.T, (o - c))
        d1 = d + np.sqrt(delta)
        d2 = d - np.sqrt(delta)
        return [d1, d2]


def projection(vector, focal_len):
    """Projects a 3D vector to image plane in 2D"""
    return vector[0:2] * focal_len / vector[2]


def calc_angle_between_2D_vectors(vec1, vec2):
    dot = vec2[0] * vec1[0] + vec1[1] * vec2[1]
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    angle_radians = math.atan2(det, dot)
    angle_degrees = angle_radians * 57.295779513
    return angle_degrees


def serialize_results_to_match_markers(
    result_file: str, start_frame: int, last_frame: int, frame_sync: int = 0
):
    """_summary_

    :param frame_sync: if results for left and right eye are desychronized
        you can pass number of frames you want to move results of RIGHT eye
        positive number means moving RIGHT eye results forward, negative backward
        relative to left eye results
    :type frame_sync: int
    """
    f = open(result_file, "r", newline="")
    results = json.load(f)
    f.close()

    if frame_sync != 0:
        # we need to split dicts for right and left eye
        right = []
        left = []
        for line in results:
            right.append(line["Right"])
            left.append(line["Left"])
        if frame_sync < 0:
            right = right[-frame_sync:]
            left = left[:frame_sync]
        else:
            right = right[:-frame_sync]
            left = left[frame_sync:]
        start_frame -= 100
        last_frame -= 100
        results = [{"Right": r, "Left": l} for r, l in zip(right, left)]

    markers_results = []
    marker_result = []
    eye_closed_counter = 0

    for i, line in enumerate(results[start_frame:last_frame]):
        # if radiuses are not 0 then for both eyes pupil is estimated
        if line["Right"]["eye radius"] != 0 and line["Left"]["eye radius"] != 0:
            marker_result.append(line)
            eye_closed_counter = 0
        else:
            eye_closed_counter += 1
            # if pupil is estimated for at least 30 consecutive frames it is counted as intentional staring at marker
            # but to eliminate random lone lines of not detected pupil, only when pupil is not detected
            # for at least 5 frames in a row marker is counted as finished
            if eye_closed_counter >= 15:
                if len(marker_result) > 100:
                    # print(f"i->{i+start_frame}")
                    # print(len(marker_result))
                    markers_results.append(marker_result)
                    marker_result = []
                    eye_closed_counter = 0

    return markers_results


def get_rotation_matrix_between_2_vectors(v1, v2):
    v1 = (v1 / np.linalg.norm(v1)).reshape(3)  # normalize v1
    v2 = (v2 / np.linalg.norm(v2)).reshape(3)  # normalize v2
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    k_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k_mat + k_mat.dot(k_mat) * ((1 - c) / (s**2))
    return rotation_matrix


def get_rotation_matrixes_between_screen_and_camera(cam_vectors, screen_vectors):
    """_summary_

    :param vectors1: _description_
    :type vectors1: _type_
    :param vectors2: _description_
    :type vectors2: _type_
    """
    rot_matrixes = []
    # distances = []  # norms of vectors from eye to screen
    for v1, v2 in zip(cam_vectors, screen_vectors):
        rot_matrixes.append(get_rotation_matrix_between_2_vectors(v1, v2))
        # distances.append(np.linalg.norm(v2))

    # weights = 1.0 / np.array(distances)
    # weights /= np.sum(weights)  # normalize weights
    # mean_rot_mat = np.average(rot_matrixes, axis=0, weights=weights)
    # transform_mat = np.linalg.inv(mean_rot_mat)
    return rot_matrixes


def rotate_vectors(vectors, rot_mat):
    """_summary_

    :param vectors: _description_
    :type vectors: _type_
    :param rot_mat: _description_
    :type rot_mat: _type_
    :return: _description_
    :rtype: _type_
    """

    vectors = np.multiply(vectors, rot_mat)
    return vectors


def create_vectors_to_markers(
    markers_pos, dist_from_screen: int, screen_size: tuple, pupilary_dist: int
):
    """Returns array with vectors to every marker from both eyes which is (0, 0, -/+ pupilary_dist/2)

    :param markers_pos: NxM numpy array representing where are markes on screen(ex. [0.25, 0.75])
    """
    middle = np.multiply(screen_size, 0.5)
    left_eye = [middle[0] + pupilary_dist / 2, middle[1]]
    right_eye = [middle[0] - pupilary_dist / 2, middle[1]]
    r_vecs = []
    l_vecs = []

    for marker in markers_pos:
        vec = np.multiply(marker, screen_size)
        left_vec = vec - left_eye
        right_vec = vec - right_eye

        left_vec = np.append(left_vec, -dist_from_screen)
        right_vec = np.append(right_vec, -dist_from_screen)

        if len(r_vecs) > 0 or len(l_vecs) > 0:
            r_vecs = np.vstack((r_vecs, right_vec))
            l_vecs = np.vstack((l_vecs, left_vec))
        else:
            r_vecs.append(right_vec)
            l_vecs.append(left_vec)

    return r_vecs, l_vecs


def calculate_base_change_matrix(camera_vectors, screen_vectors):
    """

    :param camera_vectors: _description_
    :type camera_vectors: _type_
    :param screen_vectors: _description_
    :type screen_vectors: _type_
    :return: _description_
    :rtype: _type_
    """
    screen_vectors = screen_vectors / np.linalg.norm(screen_vectors)
    A = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            A[i * 3 + j, j * 3 : (j + 1) * 3] = camera_vectors[i]
    b = np.hstack(screen_vectors)

    return np.linalg.solve(A, b).reshape((3, 3))


def get_camera_vectors_and_mean_vectors_from_results(results):
    camera_vectors_right_means = []
    camera_vectors_left_means = []
    camera_vectors_right = []
    camera_vectors_left = []
    for idx, marker in enumerate(results):
        r_vecs = []
        l_vecs = []
        for i in marker:
            r_vecs.append(i["Right"]["eye gaze vector"])
            l_vecs.append(i["Left"]["eye gaze vector"])

        camera_vectors_right.append(r_vecs)
        camera_vectors_left.append(l_vecs)

        camera_vectors_right_means.append(np.mean(r_vecs, axis=0))
        camera_vectors_left_means.append(np.mean(l_vecs, axis=0))

    return (
        camera_vectors_right,
        camera_vectors_left,
        camera_vectors_right_means,
        camera_vectors_left_means,
    )


def calculate_point_of_sight(vec, origin, dist):
    sight = vec
    sight = (-dist / sight[2]) * sight
    sight += origin
    return sight.round(3)


def get_angular_error(v1, v2):
    v1 = (v1 / np.linalg.norm(v1)).reshape(3)  # normalize v1
    v2 = (v2 / np.linalg.norm(v2)).reshape(3)  # normalize v2
    return np.arccos(np.dot(v1, v2)) * 180 / np.pi


