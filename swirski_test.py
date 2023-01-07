import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import torch
from eye_model import EyeModeling
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import utils
from models.ifOpened import ifOpenedModel
from models.trainers import PupilSegmentationTrainer, IfOpenedTrainer
from visualization import visualise_pupil
import cv2

PATH = "datasets/PupilCoreDataset/"

DATASET_LEN_TO_USE = 5000
dataset = PupilCoreDataset(
    f"{PATH}video5_eye0_video.avi",
    f"{PATH}video5_eye0_pupildata.csv",
    f"{PATH}video5_eye1_video.avi",
    f"{PATH}video5_eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)


dataset.load_masks(
    f"{PATH}created_masks/eye0",
    f"{PATH}created_masks/eye1",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ifOpenedModel = ifOpenedModel()
PupilSegmentationModel = fcn_resnet50(weights=None, num_classes=1)

if_opened_trainer = IfOpenedTrainer(
    model=ifOpenedModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)
pupil_trainer = PupilSegmentationTrainer(
    model=PupilSegmentationModel, dataset=dataset, dataset_len=DATASET_LEN_TO_USE
)

pupil_trainer.model.load_state_dict(torch.load("models/weights/resnet50.pt"))

estimated_pupil_radius_in_px = utils.get_pupil_radius_from_masks(dataset.eye0_masks)

# TODO swirski test
focal_len = 140  # ??? taken from https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py Pupil Cam3
image_shape = (192, 192)

# estiamte eye radius
# promien oka 12,5 mm
# promien oka do rogówki na ktorej żrenica byłaby dyskiem 10,5 mm
# przeliczam to na piksele skalujac z wielkoscia źrenicy w pixelach
# eye_radius/eye_radius_in_px = pupil_radius/pupil_radius_in_px
print(estimated_pupil_radius_in_px)
eye_radius_in_px = estimated_pupil_radius_in_px * 10.5 / 4

inital_eye_center_z = estimated_pupil_radius_in_px * 25.5

print(f"eye_radius_in_px: {eye_radius_in_px}")
eye_modeling = EyeModeling(
    focal_len=focal_len,
    pupil_radius=estimated_pupil_radius_in_px,
    image_shape=image_shape,
    inital_z=inital_eye_center_z,
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
            outputs_sig = np.transpose(outputs_sig.cpu().numpy(), (1, 2, 0)).copy()

            ellipse = utils.fit_ellipse(outputs_sig)
            if ellipse:
                (
                    unprojected_vectors,
                    unprojected_centers,
                ) = eye_modeling.two_circle_unprojection(ellipse)

                if i < 20:
                    image = visualise_pupil.draw_normal_vectors_2D(
                        image,
                        eye_modeling.disc_centers[-1][0][0:2],
                        eye_modeling.disc_normals[-1][0][0:2],
                        color=(0, 0, 255),
                    )
                    image = visualise_pupil.draw_normal_vectors_2D(
                        image,
                        eye_modeling.disc_centers[-1][1][0:2],
                        eye_modeling.disc_normals[-1][1][0:2],
                        color=(255, 0, 0),
                    )
                    plt.imshow(image)
                    plt.show()

        eye_modeling.sphere_centre_estimate()

    print(f"estimated 2D eye center: {eye_modeling.estimated_eye_center_2D}")
    print(f"estimated 3D eye center: {eye_modeling.estimated_eye_center_3D}")

    image = dataset.eye0_frames[0]
    visualise_pupil.draw_point(image, eye_modeling.estimated_eye_center_2D)
    visualise_pupil.draw_point(image, eye_modeling.estimated_eye_center_3D[0:2])
    plt.imshow(image)
    plt.show()

    radius = eye_modeling.sphere_radius_estimate()

    for i, (inputs, masks, opened) in enumerate(pupil_trainer.dataloaders["test"]):

        print("--------------------------------------------")
        if opened:
            inputs = inputs.to(device)
            outputs = pupil_trainer.model(inputs)

            image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
            outputs_sig = torch.sigmoid(outputs["out"][0])
            outputs_sig = np.transpose(outputs_sig.cpu().numpy(), (1, 2, 0)).copy()

            ellipse = utils.fit_ellipse(outputs_sig)
            if ellipse:
                (
                    unprojected_vectors,
                    unprojected_centers,
                ) = eye_modeling.two_circle_unprojection(ellipse)

                unprojected_centers_list = []
                unprojected_vectors_list = []
                unprojected_centers_list.append(unprojected_centers)
                unprojected_vectors_list.append(unprojected_vectors)

                (
                    filtered_vector,
                    filtered_pos,
                ) = eye_modeling.filter_vectors_towards_center(
                    unprojected_vectors_list, unprojected_centers_list
                )
                # print(filtered_pos)
                if filtered_pos:
                    try:
                        (
                            pupil_pos,
                            pupil_normal,
                            pupil_radius,
                        ) = eye_modeling.consistent_pupil_estimate(
                            np.array(filtered_pos).ravel().reshape(3, 1)
                        )
                        print(
                            f" new pupil: position-{pupil_pos}, normal_vector-{pupil_normal}, pupil_radius-{pupil_radius}"
                        )
                    except Exception as err:
                        print(err)
                else:
                    print("żaden z wektorów nie pokazuje od środka oka")

        if i > 5:
            break
