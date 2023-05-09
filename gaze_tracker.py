import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import torch
from eye_model import EyeModeling
from exceptions import NoIntersection, NoEllipseFound
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from visualization import visualise_pupil
from datasets.utils import get_one_dataloader
import utils
import numpy as np
import csv
import warnings
import time
import json

warnings.filterwarnings("error")


class GazeTracker:
    def __init__(self, weight_path: str, px_to_mm: int = 167) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_if_opened = ifOpenedModel()
        self.cnn_if_opened.load_state_dict(
            torch.load(f"{weight_path}/squeeznet1_1.pt", map_location=self.device)
        )
        self.cnn_pupil_segmentation = pupilSegmentationModel()
        self.cnn_pupil_segmentation.load_state_dict(
            torch.load(f"{weight_path}/resnet50.pt", map_location=self.device)
        )
        self.cnn_pupil_segmentation = self.cnn_pupil_segmentation.to(self.device)
        self.cnn_if_opened = self.cnn_if_opened.to(self.device)
        self.MMTOPX = px_to_mm
        self.estimated_pupil_radius_in_px = 2 * self.MMTOPX
        self.inital_eye_center_z = 35 * self.MMTOPX

    def filter_ellipse(self, ellipse) -> bool:
        """If ellipse comes mathces filter it is good to estimate eye model"""
        if (
            abs(ellipse[1][0] - ellipse[1][1]) > 2.5
            and abs(ellipse[1][0] - ellipse[1][1]) < 5
        ):
            return True
        else:
            return False

    def draw_current_pupil_vectors(self, image, eye_model):
        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_model.ellipses[-1][0:2],
            # eye_model.disc_centers[-1][0][0:2].ravel(),
            eye_model.disc_normals[-1][0][0:2].ravel(),
            color=(255, 0, 0),
        )
        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_model.ellipses[-1][0:2],
            # eye_model.disc_centers[-1][0][0:2].ravel(),
            eye_model.disc_normals[-1][1][0:2].ravel(),
            color=(0, 255, 0),
        )

        return image

    def fit_tracker(self, dataset: PupilCoreDataset):
        dataloader = get_one_dataloader(dataset)
        image_shape = dataset.image_shape[:2]
        self.right_eye_modeling = EyeModeling(
            focal_len=dataset.focal_len,
            pupil_radius=self.estimated_pupil_radius_in_px,
            image_shape=image_shape,
            inital_z=self.inital_eye_center_z,
        )
        self.left_eye_modeling = EyeModeling(
            focal_len=dataset.focal_len,
            pupil_radius=self.estimated_pupil_radius_in_px,
            image_shape=image_shape,
            inital_z=self.inital_eye_center_z,
        )
        self.eyes_models = [self.right_eye_modeling, self.left_eye_modeling]

        with torch.no_grad():
            self.cnn_pupil_segmentation.eval()
            self.cnn_if_opened.eval()
            base_right_image = dataset.eye0_frames[0].copy()
            base_left_image = dataset.eye1_frames[0].copy()
            for i, (inputs, opened) in enumerate(dataloader):
                # iterate though both eyes
                # 0(first) is right eye, 1(second) left eye
                for x in range(2):
                    eye_model = self.eyes_models[x]
                    _input = inputs[x].to(self.device)
                    outputs = self.cnn_if_opened(_input)
                    _, preds = torch.max(outputs, 1)

                    if preds[0]:
                        outputs = self.cnn_pupil_segmentation(_input)
                        outputs_sig = torch.sigmoid(outputs["out"][0])
                        outputs_sig = np.transpose(
                            outputs_sig.cpu().numpy(), (1, 2, 0)
                        ).copy()
                        try:
                            ellipse = utils.fit_ellipse(outputs_sig)
                            # ellipse = dataset.get_ellipses_from_data(i, x)
                            # and self.filter_ellipse(ellipse)
                            if ellipse:
                                (
                                    unprojected_vectors,
                                    unprojected_centers,
                                ) = eye_model.two_circle_unprojection(ellipse)

                                angle = utils.calc_angle_between_2D_vectors(
                                    unprojected_vectors[0][0:2],
                                    unprojected_vectors[1][0:2],
                                )

                                # this part is for visalization
                                # ============================================================

                                # if i % 100 == 0:

                                #     if x == 0:
                                #         base_right_image = visualise_pupil.draw_ellipse(
                                #             base_right_image,
                                #             ellipse,
                                #             (0, 0, 255),
                                #         )
                                #         base_right_image = (
                                #             self.draw_current_pupil_vectors(
                                #                 image=base_right_image,
                                #                 eye_model=eye_model,
                                #             )
                                #         )

                                #     else:
                                #         base_left_image = visualise_pupil.draw_ellipse(
                                #             base_left_image,
                                #             ellipse,
                                #             (0, 0, 255),
                                #         )
                                #         base_left_image = (
                                #             self.draw_current_pupil_vectors(
                                #                 image=base_left_image,
                                #                 eye_model=eye_model,
                                #             )
                                #         )

                                # print(i)
                                # image = np.transpose(
                                #     _input[0].cpu().numpy(), (1, 2, 0)
                                # ).copy()
                                # image = visualise_pupil.draw_ellipse(
                                #     image, ellipse, (0, 0, 255)
                                # )

                                # ellipse_from_data = dataset.get_ellipses_from_data(
                                #     i, x
                                # )
                                # image = visualise_pupil.draw_ellipse(
                                #     image, ellipse_from_data, (255, 0, 255)
                                # )
                                # print(ellipse)
                                # print(ellipse_from_data)
                                # plt.imshow(dataset.eye0_masks[i])
                                # plt.show()
                                # image = self.draw_current_pupil_vectors(
                                #     image=image, eye_model=eye_model
                                # )
                                # plt.imshow(image)
                                # plt.show()
                                # ===========================================================
                        except NoEllipseFound as err:
                            print(err.message)

        for eye in self.eyes_models:
            eye.sphere_centre_estimate()
            eye.sphere_radius_estimate()
            print(f"estimated eye center in 2D -> {eye.estimated_eye_center_2D}")
            print(f"estimated eye center in 3D -> {eye.estimated_eye_center_3D}")
            print(f"estimated eye radius -> {eye.estimated_sphere_radius}")

        # base_right_image = visualise_pupil.draw_point(
        #     base_right_image, self.eyes_models[0].estimated_eye_center_2D
        # )
        # base_left_image = visualise_pupil.draw_point(
        #     base_left_image, self.eyes_models[1].estimated_eye_center_2D
        # )
        # plt.imshow(base_right_image)
        # plt.show()
        # plt.imshow(base_left_image)
        # plt.show()

    def track_gaze_vector(
        self,
        dataset: PupilCoreDataset,
        result_save_path: str = "results/default_path.csv",
    ):
        dataloader = get_one_dataloader(dataset)
        for eye in self.eyes_models:
            eye.disc_normals = []
            eye.disc_centers = []

        f = open(file=result_save_path, mode="w", newline="")
        rows = []

        with torch.no_grad():
            for i, (inputs, opened) in enumerate(dataloader):
                row = {}
                # iterate though both eyes
                # 0(first) is right eye, 1(second) left eye
                for x in range(2):
                    eye_model = self.eyes_models[x]
                    _side = "Right" if x == 0 else "Left"

                    _input = inputs[x].to(self.device)
                    outputs = self.cnn_if_opened(_input)
                    _, preds = torch.max(outputs, 1)

                    # if i % 50 == 0:
                    #     image = np.transpose(_input[0].cpu().numpy(), (1, 2, 0)).copy()
                    #     plt.imshow(image)
                    #     plt.show()
                    #     print(i)
                    empty_pupil_result = {
                        "Index": i,
                        _side: {
                            "eye pupil position": [0, 0, 0],
                            "eye gaze vector": [0, 0, 0],
                            "eye radius": 0,
                            "eye opened": (preds.cpu().tolist())[0],
                        },
                    }

                    if preds[0]:
                        try:
                            outputs = self.cnn_pupil_segmentation(_input)
                            outputs_sig = torch.sigmoid(outputs["out"][0])
                            outputs_sig = np.transpose(
                                outputs_sig.cpu().numpy(), (1, 2, 0)
                            ).copy()
                            ellipse = utils.fit_ellipse(outputs_sig)

                            if ellipse:
                                (
                                    unprojected_vectors,
                                    unprojected_centers,
                                ) = eye_model.two_circle_unprojection(ellipse)
                                (
                                    filtered_vector,
                                    filtered_pos,
                                ) = eye_model.filter_vectors_towards_center(
                                    [unprojected_vectors], [unprojected_centers]
                                )
                                # this part is for visalization
                                # # ============================================================
                                # print("--------------------------------------------")
                                # print(f"ellipse -> {ellipse}")
                                # image = np.transpose(
                                #     _input[0].cpu().numpy(), (1, 2, 0)
                                # ).copy()
                                # image = visualise_pupil.draw_point(
                                #     image, eye_model.estimated_eye_center_2D
                                # )
                                # image = self.draw_current_pupil_vectors(
                                #     image=image, eye_model=eye_model
                                # )
                                # image = visualise_pupil.draw_ellipse(
                                #     image, ellipse, (0, 0, 255)
                                # )
                                # plt.imshow(image)
                                # plt.show()
                                # =============================================================
                                if filtered_pos:
                                    try:
                                        (
                                            pupil_pos,
                                            pupil_normal,
                                            pupil_radius,
                                        ) = eye_model.consistent_pupil_estimate(
                                            np.array(filtered_pos).ravel().reshape(3, 1)
                                        )

                                        row.update(
                                            {
                                                "Index": i,
                                                _side: {
                                                    f"eye pupil position": pupil_pos.ravel()
                                                    .round(3)
                                                    .tolist(),
                                                    f"eye gaze vector": pupil_normal.ravel()
                                                    .round(3)
                                                    .tolist(),
                                                    f"eye radius": pupil_radius[
                                                        0
                                                    ].round(3),
                                                    f"eye opened": (
                                                        preds.cpu().tolist()
                                                    )[0],
                                                },
                                            }
                                        )
                                        # This part is for visualization
                                        # if i % 100 == 0:
                                        #     print(
                                        #         f" new pupil: position-{pupil_pos}, normal_vector-{pupil_normal}, pupil_radius-{pupil_radius}"
                                        # )
                                    except NoIntersection as err:
                                        print(err.message)
                                        row.update(empty_pupil_result)
                                    except Exception:
                                        print(err)
                            else:
                                row.update(empty_pupil_result)
                                print("żaden z wektorów nie pokazuje od środka oka")
                        except NoEllipseFound as err:
                            print(err.message)
                            row.update(empty_pupil_result)
                    else:
                        row.update(empty_pupil_result)
                rows.append(row)

        json.dump(rows, f)
        f.close()
