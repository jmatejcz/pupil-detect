import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import torch
from eye_model import EyeModeling
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
import numpy as np
import utils
from models.ifOpened import ifOpenedModel
from models.trainers import PupilSegmentationTrainer, IfOpenedTrainer
from visualization import visualise_pupil
import cv2


DATASET_LEN_TO_USE = 5000
dataset = PupilCoreDataset(
    "datasets/PupilCoreDataset/video5_eye0_video.avi",
    "datasets/PupilCoreDataset/video5_eye0_pupildata.csv",
    "datasets/PupilCoreDataset/video5_eye1_video.avi",
    "datasets/PupilCoreDataset/video5_eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.load_masks(
    "datasets/PupilCoreDataset/created_masks/eye0",
    "datasets/PupilCoreDataset/created_masks/eye1",
)

ifOpenedModel = ifOpenedModel()
PupilSegmentationModel = fcn_resnet50(weights=None, num_classes=1)

if_opened_trainer = IfOpenedTrainer(
    model=ifOpenedModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)
pupil_trainer = PupilSegmentationTrainer(
    model=PupilSegmentationModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)

pupil_trainer.model.load_state_dict(torch.load("models/weights/resnet50.pt"))

estimate_radius_in_px = utils.get_pupil_radius_from_masks(dataset.eye0_masks)

# TODO swirski test
focal_len = 140  # ??? taken from https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py Pupil Cam3
image_shape = (192, 192)
inital_eye_center_z = estimate_radius_in_px*40
eye_modeling = EyeModeling(
    focal_len=focal_len, pupil_radius=estimate_radius_in_px, image_shape=image_shape, inital_z=inital_eye_center_z
)

with torch.no_grad():
    pupil_trainer.model.eval()
    pupil_trainer.model = pupil_trainer.model.to(device)
    for i, (inputs, masks, opened) in enumerate(pupil_trainer.dataloaders["test"]):

        if opened:
            inputs = inputs.to(device)
            outputs = pupil_trainer.model(inputs)

            image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
            outputs_sig = torch.sigmoid(outputs["out"][0])
            outputs_sig = np.transpose(
                outputs_sig.cpu().numpy(), (1, 2, 0)).copy()

            ellipse = utils.fit_ellipse(outputs_sig)
            if ellipse:
                eye_modeling.two_circle_unprojection(ellipse)

            # if i > 50:

        #     break

    eye_modeling.sphere_centre_estimate()
    print(eye_modeling.estimated_eye_center_2D)
    image = dataset.eye0_frames[0]
    for i in range(10):

        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_modeling.disc_centers[i][0][0:2],
            eye_modeling.disc_normals[i][0][0:2],
            color=(0, 0, 255),
        )
        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_modeling.disc_centers[i][1][0:2],
            eye_modeling.disc_normals[i][1][0:2],
            color=(255, 0, 0),
        )
    plt.imshow(image)
    plt.show()

    radius = eye_modeling.sphere_radius_estimate()
    print(radius)
