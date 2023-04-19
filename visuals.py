import cv2
from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import numpy as np
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from torchvision import models
import torch
import matplotlib.pyplot as plt
from datasets.utils import get_one_dataloader
from visualization import visualise_pupil
import utils


# ==========================================================
# file for generating images for thesis

PATH = "datasets/PupilCoreDataset/"

dataset = PupilCoreDatasetGazeTrack(
    f"{PATH}/1/eye0_video.avi",
    f"{PATH}/1/eye0_pupildata.csv",
    f"{PATH}/1/eye1_video.avi",
    f"{PATH}/1/eye1_pupildata.csv",
    dataset_len=500,
)
path_to_save = "visualization/images/"

cnn_if_opened = ifOpenedModel()
cnn_if_opened.load_state_dict(torch.load("models/weights/squeeznet1_1.pt"))

cnn_pupil_segmentation = pupilSegmentationModel()
cnn_pupil_segmentation.load_state_dict(torch.load("models/weights/resnet50.pt"))
# print(model)
# print(cnn_if_opened)
dataloader = get_one_dataloader(dataset)

# for i, (inputs, opened) in enumerate(dataset):

#     if i % 100 == 0:
#         image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
#         #     model.eval()
#     fig = plt.figure()


def if_opened_image(inputs):
    outputs = cnn_if_opened(inputs[0])
    _, preds = torch.max(outputs, 1)
    image = np.transpose(inputs[0][0].numpy(), (1, 2, 0)).copy()
    image = visualise_pupil.visualize_ifopened(image, preds[0])


def save_if_opened_image(image):
    image = image * 256
    cv2.imwrite(f"{path_to_save}/ifopened_{i}.jpg", image)


def segmenation_image(inputs):
    outputs = cnn_pupil_segmentation(inputs[0])
    image = np.transpose(inputs[0][0].numpy(), (1, 2, 0)).copy()
    outputs_sig = torch.sigmoid(outputs["out"][0])
    outputs_sig = np.transpose(outputs_sig.numpy(), (1, 2, 0)).copy()
    ellipse = utils.fit_ellipse(outputs_sig)

    if ellipse:
        image = visualise_pupil.draw_ellipse(image, ellipse)


def save_segmentation_image(image):
    image = image * 256
    cv2.imwrite(f"{path_to_save}/ellipse_{i}.jpg", image)


with torch.no_grad():
    true_preds = 0
    cnn_if_opened.eval()
    for i, (inputs, opened) in enumerate(dataloader):
        ### wykrywanie mrugania obrazy#############################
        # if i % 50 == 0:
        #     outputs = cnn_if_opened(inputs[0])
        #     _, preds = torch.max(outputs, 1)
        #     image = np.transpose(inputs[0][0].numpy(), (1, 2, 0)).copy()
        #     image = visualise_pupil.visualize_ifopened(image, preds[0])
        #     image = image * 256
        #     cv2.imwrite(f"{path_to_save}/ifopened_{i}.jpg", image)
        # plt.imshow(image)
        # plt.savefig(f"{path_to_save}/ifopened_{i}")
        ############################################################

        ### accuracy ifopened ######################################
        #     label = int(opened[0])
        #     outputs = cnn_if_opened(inputs[0])
        #     _, preds = torch.max(outputs, 1)
        #     if int(preds[0]) == label:
        #         true_preds += 1
        #     else:
        #         # print(preds, label)
        #         image = np.transpose(inputs[0][0].numpy(), (1, 2, 0)).copy()
        #         image = visualise_pupil.visualize_ifopened(image, preds[0])
        #         image = image * 256
        #         # plt.imshow(image)
        #         # plt.show()
        #         cv2.imwrite(f"{path_to_save}/ifopened_bledne_{i}.jpg", image)

        # if_opened_acc = true_preds / i * 100
        # print(if_opened_acc)

        ### segmentacja visuals ################
        if i % 50 == 0:
            if opened[0]:

                outputs = cnn_pupil_segmentation(inputs[0])
                image = np.transpose(inputs[0][0].numpy(), (1, 2, 0)).copy()
                outputs_sig = torch.sigmoid(outputs["out"][0])
                # outputs_sig = torch.nn.functional.softmax(outputs["out"][0], dim=1)
                outputs_sig = np.transpose(outputs_sig.numpy(), (1, 2, 0)).copy()
                # outputs_sig = np.where(outputs_sig > 0.4, 1, 0)
                ellipse = utils.fit_ellipse(outputs_sig)

                if ellipse:
                    image = visualise_pupil.draw_ellipse(image, ellipse)
                plt.imshow(image)
                # image = image * 256

                # cv2.imwrite(f"{path_to_save}/ellipse_{i}.jpg", image)
                plt.show()

                visualise_pupil.draw_ellipse(image)
