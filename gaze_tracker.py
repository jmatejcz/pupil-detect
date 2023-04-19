import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import torch
from eye_model import EyeModeling
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from visualization import visualise_pupil
from datasets.utils import get_one_dataloader
import utils
import numpy as np


class GazeTracker:
    def __init__(self, weight_path) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_if_opened = ifOpenedModel()
        self.cnn_if_opened.load_state_dict(torch.load(
            f"{weight_path}/squeeznet1_1.pt", map_location=self.device))
        self.cnn_pupil_segmentation = pupilSegmentationModel()
        self.cnn_pupil_segmentation.load_state_dict(
            torch.load(f"{weight_path}/resnet50.pt", map_location=self.device)
        )
        self.cnn_pupil_segmentation = self.cnn_pupil_segmentation.to(
            self.device)
        self.cnn_if_opened = self.cnn_if_opened.to(self.device)

    def fit_tracker(self, dataset: PupilCoreDataset):
        dataloader = get_one_dataloader(dataset)
        dataset.get_pupil_masks()
        estimated_pupil_radius_in_px = utils.get_pupil_radius_from_masks(
            dataset.eye0_masks
        )
        # eye_radius_in_px is not used in algortihm, its just to estimate how to set inital_eye_z
        eye_radius_in_px = estimated_pupil_radius_in_px * 10.5 / 2
        inital_eye_center_z = estimated_pupil_radius_in_px * 24 * 2
        image_shape = dataset.image_shape[:2]
        self.focal_len = dataset.focal_len

        self.eye_modeling = EyeModeling(
            focal_len=self.focal_len,
            pupil_radius=estimated_pupil_radius_in_px,
            image_shape=image_shape,
            inital_z=inital_eye_center_z,
        )

        with torch.no_grad():
            self.cnn_pupil_segmentation.eval()
            self.cnn_if_opened.eval()
            for i, (inputs, opened) in enumerate(dataloader):

                inputs = inputs[0].to(self.device)
                outputs = self.cnn_if_opened(inputs)
                _, preds = torch.max(outputs, 1)
                if opened[0]:

                    outputs = self.cnn_pupil_segmentation(inputs)

                    image = np.transpose(
                        inputs[0].cpu().numpy(), (1, 2, 0)).copy()
                    outputs_sig = torch.sigmoid(outputs["out"][0])
                    outputs_sig = np.transpose(
                        outputs_sig.cpu().numpy(), (1, 2, 0)
                    ).copy()

                    ellipse = utils.fit_ellipse(outputs_sig)
                    if ellipse:

                        (
                            unprojected_vectors,
                            unprojected_centers,
                        ) = self.eye_modeling.two_circle_unprojection(ellipse)

                        # this part is for visalization
                        # ============================================================
                        # if i % 100 == 0:
                        # image = visualise_pupil.draw_ellipse(image, ellipse)
                        # image = visualise_pupil.draw_normal_vectors_2D(
                        #     image,
                        #     self.eye_modeling.ellipse_centers[-1],
                        #     self.eye_modeling.disc_normals[-1][0][0:2].ravel(),
                        #     color=(0, 0, 255),
                        # )
                        # image = visualise_pupil.draw_normal_vectors_2D(
                        #     image,
                        #     self.eye_modeling.ellipse_centers[-1],
                        #     self.eye_modeling.disc_normals[-1][1][0:2].ravel(),
                        #     color=(255, 0, 0),
                        # )
                        # plt.imshow(image)
                        # plt.show()
                        # ===========================================================
            self.eye_modeling.sphere_centre_estimate()

        self.eye_modeling.sphere_radius_estimate()
        print(
            f"estimated 2D eye center -> {self.eye_modeling.estimated_eye_center_2D}")
        print(
            f"estimated 3D eye center -> {self.eye_modeling.estimated_eye_center_3D}")
        print(
            f"estimated eye radius -> {self.eye_modeling.estimated_sphere_radius}")
        print(f"avarage eye radius in px -> {eye_radius_in_px}")
        print(f"average pupil radius in px -> {estimated_pupil_radius_in_px}")

    def track_gaze_vector(self, dataset: PupilCoreDataset):
        dataloader = get_one_dataloader(dataset)
        self.eye_modeling.disc_normals = []
        self.eye_modeling.disc_centers = []

        with torch.no_grad():
            for i, (inputs, opened) in enumerate(dataloader):

                print("--------------------------------------------")
                if opened[0]:
                    inputs = inputs[0].to(self.device)
                    outputs = self.cnn_pupil_segmentation(inputs)

                    outputs_sig = torch.sigmoid(outputs["out"][0])
                    outputs_sig = np.transpose(
                        outputs_sig.cpu().numpy(), (1, 2, 0)
                    ).copy()

                    ellipse = utils.fit_ellipse(outputs_sig)
                    if ellipse:
                        (
                            unprojected_vectors,
                            unprojected_centers,
                        ) = self.eye_modeling.two_circle_unprojection(ellipse)
                        (
                            filtered_vector,
                            filtered_pos,
                        ) = self.eye_modeling.filter_vectors_towards_center(
                            [unprojected_vectors], [unprojected_centers]
                        )
                        # this part is for visalization
                        # ============================================================
                        image = np.transpose(
                            inputs[0].cpu().numpy(), (1, 2, 0)).copy()
                        image = visualise_pupil.draw_point(
                            image, self.eye_modeling.estimated_eye_center_3D[:2]
                        )
                        image = visualise_pupil.draw_normal_vectors_2D(
                            image,
                            unprojected_centers[0][0:2],
                            unprojected_vectors[0][0:2],
                            color=(255, 0, 0),
                        )
                        image = visualise_pupil.draw_normal_vectors_2D(
                            image,
                            unprojected_centers[1][0:2],
                            unprojected_vectors[1][0:2],
                            color=(0, 0, 255),
                        )
                        plt.imshow(image)
                        plt.show()
                        # =============================================================
                        if filtered_pos:
                            try:
                                (
                                    pupil_pos,
                                    pupil_normal,
                                    pupil_radius,
                                ) = self.eye_modeling.consistent_pupil_estimate(
                                    np.array(
                                        filtered_pos).ravel().reshape(3, 1)
                                )
                                print(
                                    f" new pupil: position-{pupil_pos}, normal_vector-{pupil_normal}, pupil_radius-{pupil_radius}"
                                )

                                image = np.transpose(
                                    inputs[0].cpu().numpy(), (1, 2, 0)
                                ).copy()
                                image = visualise_pupil.draw_point(
                                    image, self.eye_modeling.estimated_eye_center_3D[:2]
                                )
                                image = visualise_pupil.draw_normal_vectors_2D(
                                    image,
                                    pupil_pos[0:2],
                                    pupil_normal[0:2],
                                    color=(255, 0, 0),
                                )
                                plt.imshow(image)
                                plt.show()
                            except Exception as err:
                                print(err)
                        else:
                            print("żaden z wektorów nie pokazuje od środka oka")

                if i > 10:
                    break
