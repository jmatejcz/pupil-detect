import time
import copy
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


def visualize_ifopened(model, device, dataloaders, num_images=6, class_to_show=None):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            label = int(labels[0])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
            x, y = image.shape[0], image.shape[1]
            digit = int(preds[0])
            # if class_to_show:
            #     cv2.putText(image, f"{digit}", (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            #     ax = plt.imshow(image)
            if label == class_to_show:
                cv2.putText(
                    image,
                    f"{digit}",
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                )
                ax = plt.imshow(image)

            if i >= num_images:
                model.train(mode=was_training)
                return

        model.train(mode=was_training)


def visualize_pupil(input_img, output_img):

    image = np.transpose(input_img.cpu().numpy(), (1, 2, 0)).copy()
    outputs_sig = torch.sigmoid(output_img)
    outputs_sig = np.transpose(outputs_sig.cpu().detach().numpy(), (1, 2, 0)).copy()

    plt.imshow(image)
    plt.show()

    plt.imshow(outputs_sig)
    plt.show()


def evaluate_pupil(model, dataloaders, device):
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            print(outputs, labels)
            diff = (labels[0][0] - outputs[0][0], labels[0][1] - outputs[0][1])
            print(diff)


def fit_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    thresh = np.asarray(thresh, dtype=np.uint8)

    countours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    # return sorted_


def calc_intersection(normal_vectors_2D, centers_2D):
    """Calculate point nearest to all the lines, starting in centers

    :param normal_vectors_2D: _description_
    :type normal_vectors_2D: _type_
    :param centers_2D: _description_
    :type centers_2D: _type_
    """
    R_sum = 0
    S_sum = 0
    print(normal_vectors_2D[0].shape)
    # TODO normalizacja wektorów?
    I = np.eye(normal_vectors_2D[0].shape[1])
    for i, vec in enumerate(normal_vectors_2D):
        # print(np.matmul(vec, vec.transpose()))
        R = I - np.matmul(vec, vec.transpose())
        # print(f"R : {R}")
        R_sum += R
        S = np.matmul(R, centers_2D[i])
        # print(f"S : {S}")
        S_sum += S

    # print(R_sum, S_sum)

    c = np.matmul(np.linalg.inv(R_sum), S_sum)

    return c
