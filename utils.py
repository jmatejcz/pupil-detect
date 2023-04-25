import cv2
import numpy as np
from exceptions import NoEllipseFound, NoIntersection


def fit_ellipse(mask):
    _, thresh = cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY)
    thresh = np.asarray(thresh, dtype=np.uint8)

    countours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = [i[0] for i in countours[0]]
        cnt = np.asarray(cnt)
        ellipse = cv2.fitEllipse(cnt)
        # catch all weird invalid shapes
        if (
            ellipse[2] == 90.0
            or ellipse[2] == 0.0
            or ellipse[1][0] == 1.0
            or ellipse[1][0] == 1.0
        ):
            raise NoEllipseFound("Invalid ellipse shape")
        else:
            return cv2.fitEllipse(cnt)
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
