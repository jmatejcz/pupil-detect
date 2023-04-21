import time
import copy
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


def evaluate_pupil(model, dataloaders, device):
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            diff = (labels[0][0] - outputs[0][0], labels[0][1] - outputs[0][1])


def fit_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    thresh = np.asarray(thresh, dtype=np.uint8)

    countours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(countours)
    try:
        # if len(countours) > 0:
        cnt = [i[0] for i in countours[0]]
        cnt = np.asarray(cnt)
        return cv2.fitEllipse(cnt)
    except Exception as err:
        pass


def draw_ellipse(image, ellipse):
    cv2.ellipse(image, ellipse, (0, 0, 255), 1)
    return image


def get_pupil_radius_from_masks(masks):
    """Get mean pupil radius in pixel from masks
    Masks should be with pupil looking straight to the camera

    :param images: _description_
    :type images: _type_
    """
    ellipses = []
    for mask in masks:
        ellipse = fit_ellipse(mask)
        if ellipse:
            axis = ellipse[1]
            ellipses.append((ellipse, axis[1] - axis[0]))

    sorted_ = sorted(ellipses, key=lambda x: x[1])[:50]
    radiuses = []
    for el in sorted_:
        radiuses.append(el[0][1][0] / 2)

    return np.mean(radiuses)


def calc_intersection(vectors, centers):
    """
    Calculate point nearest to all the lines, starting in centers
    """
    R_sum = 0
    S_sum = 0
    # TODO normalizacja wektorów?
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
    # print(u)
    delta = np.square(np.dot(u.T, (o - c))) - \
        np.dot((o - c).T, (o - c)) + np.square(r)
    if delta < 0:
        print("delta less than 0")
    else:
        d = -np.dot(u.T, (o - c))
        # print(d, delta)
        d1 = d + np.sqrt(delta)
        d2 = d - np.sqrt(delta)
        return [d1, d2]


def calc_line_plane_intersection(vec, o, plane):
    """_summary_

    :param vec: vector
    :type vec: _type_
    :param o: origin
    :type o: _type_
    :param plane: 3 points of a plane [A, B, C]
    :type plane: _type_
    """


def convert_to_px(value):
    """converts values in milimeters to pixels, depending on 

    :param value: _description_
    :type value: _type_
    """
    pass
