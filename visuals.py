import cv2
from datasets.PupilCoreDatasetPupil import PupilCoreDatasetGazeTrack
import numpy as np
from models.ifOpened import ifOpenedModel
from torchvision import models

# ==========================================================
# file for generating images for thesis

PATH = "datasets/PupilCoreDataset/"

# dataset = PupilCoreDatasetGazeTrack(
#     f"{PATH}video5_eye0_video.avi",
#     f"{PATH}video5_eye0_pupildata.csv",
#     f"{PATH}video5_eye1_video.avi",
#     f"{PATH}video5_eye1_pupildata.csv",
#     dataset_len=5000,
# )
path_to_save = "visualization/images/"

cnn_if_opened = ifOpenedModel()
model = models.squeezenet1_1()
print(model)
print(cnn_if_opened)

# for i, (inputs, opened) in enumerate(dataset):

#     if i % 100 == 0:
#         image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
#         image = image * 256
#         # cv2.imshow("xd", image)
#         cv2.imwrite(f"visualization/images/{i}.jpg", image)
