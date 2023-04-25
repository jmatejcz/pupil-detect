import matplotlib.pyplot as plt
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import torch
from eye_model import EyeModeling
from exceptions import NoIntersection
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from visualization import visualise_pupil
from datasets.utils import get_one_dataloader
import utils
import numpy as np
import csv


class GazeTracker:
    def __init__(self, weight_path) -> None:
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

    def fit_tracker(self, dataset: PupilCoreDataset):
        dataloader = get_one_dataloader(dataset)
        dataset.get_pupil_masks()
        estimated_pupil_radius_in_px = utils.get_pupil_radius_from_masks(
            dataset.eye0_masks
        )
        MMTOPX = 30
        estimated_pupil_radius_in_px = 2 * MMTOPX
        # eye_radius_in_px is not used in algortihm, its just to estimate how to set inital_eye_z
        # print(f"scaling -> {np.array([192, 192]) / np.array([3.6, 4.8])}")
        eye_radius_in_px = estimated_pupil_radius_in_px * 10.5 * MMTOPX
        inital_eye_center_z = 51 * MMTOPX
        image_shape = dataset.image_shape[:2]
        self.focal_len = dataset.focal_len

        self.right_eye_modeling = EyeModeling(
            focal_len=self.focal_len,
            pupil_radius=estimated_pupil_radius_in_px,
            image_shape=image_shape,
            inital_z=inital_eye_center_z,
        )
        self.left_eye_modeling = EyeModeling(
            focal_len=self.focal_len,
            pupil_radius=estimated_pupil_radius_in_px,
            image_shape=image_shape,
            inital_z=inital_eye_center_z,
        )
        self.eyes_models = [self.right_eye_modeling, self.left_eye_modeling]
        with torch.no_grad():
            self.cnn_pupil_segmentation.eval()
            self.cnn_if_opened.eval()
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
                        ellipse = utils.fit_ellipse(outputs_sig)

                        if ellipse:
                            eye_model.two_circle_unprojection(ellipse)
                            # this part is for visalization
                            # ============================================================
                            if i % 100 == 0:
                                print(f"ellipsa -> {ellipse}")
                                print(x)
                                image = np.transpose(
                                    _input[0].cpu().numpy(), (1, 2, 0)
                                ).copy()
                                image = visualise_pupil.draw_ellipse(image, ellipse)
                                image = visualise_pupil.draw_normal_vectors_2D(
                                    image,
                                    eye_model.ellipse_centers[-1],
                                    eye_model.disc_normals[-1][0][0:2].ravel(),
                                    color=(0, 0, 255),
                                )
                                image = visualise_pupil.draw_normal_vectors_2D(
                                    image,
                                    eye_model.ellipse_centers[-1],
                                    eye_model.disc_normals[-1][1][0:2].ravel(),
                                    color=(255, 0, 0),
                                )
                                plt.imshow(image)
                                plt.show()
                            # ===========================================================
        for eye in self.eyes_models:
            eye.sphere_centre_estimate()
            eye.sphere_radius_estimate()
            print(eye.estimated_eye_center_2D)
            print(eye.estimated_eye_center_3D)
            print(eye.estimated_sphere_radius)

    def track_gaze_vector(
        self,
        dataset: PupilCoreDataset,
        result_save_path: str = "results/default_path.csv",
    ):
        dataloader = get_one_dataloader(dataset)
        for eye in self.eyes_models:
            eye.disc_normals = []
            eye.disc_centers = []

        f = open(file=result_save_path, mode="w")
        # add headers
        headers = [
            "Index",
            "Right eye pupil position",
            "Right eye gaze vector",
            "Right eye radius",
            "Right eye opened",
            "Left eye pupil position",
            "Left eye gaze vector",
            "Left eye radius",
            "Left eye opened",
        ]
        writer = csv.DictWriter(f, delimiter=",", fieldnames=headers)
        writer.writeheader()
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

                    if preds[0]:
                        outputs = self.cnn_pupil_segmentation(_input)
                        outputs_sig = torch.sigmoid(outputs["out"][0])
                        outputs_sig = np.transpose(
                            outputs_sig.cpu().numpy(), (1, 2, 0)
                        ).copy()
                        ellipse = utils.fit_ellipse(outputs_sig)

                        empty_pupil_result = {
                            "Index": i,
                            f"{_side} eye pupil position": np.array([0, 0, 0]),
                            f"{_side} eye gaze vector": np.array([0, 0, 0]),
                            f"{_side} eye radius": 0,
                            f"{_side} eye opened": (preds.cpu().numpy())[0],
                        }

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
                            # ============================================================
                            # print("--------------------------------------------")
                            # print(f"ellipsa -> {ellipse}")
                            # print(f"index in dataset -> {i}")
                            # print(unprojected_centers[0], unprojected_vectors[0])
                            # print(unprojected_centers[1], unprojected_vectors[1])
                            # image = np.transpose(
                            #     _input.cpu().numpy(), (1, 2, 0)
                            # ).copy()
                            # image = visualise_pupil.draw_point(
                            #     image, eye_model.estimated_eye_center_2D
                            # )
                            # image = visualise_pupil.draw_normal_vectors_2D(
                            #     image,
                            #     unprojected_centers[0][0:2],
                            #     unprojected_vectors[0][0:2],
                            #     color=(255, 0, 0),
                            # )
                            # image = visualise_pupil.draw_normal_vectors_2D(
                            #     image,
                            #     unprojected_centers[1][0:2],
                            #     unprojected_vectors[1][0:2],
                            #     color=(0, 0, 255),
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
                                            f"{_side} eye pupil position": pupil_pos.ravel().round(
                                                3
                                            ),
                                            f"{_side} eye gaze vector": pupil_normal.ravel().round(
                                                3
                                            ),
                                            f"{_side} eye radius": pupil_radius[
                                                0
                                            ].round(3),
                                            f"{_side} eye opened": (
                                                preds.cpu().numpy()
                                            )[0],
                                        }
                                    )

                                    # This part is for visualization
                                    # if i % 100 == 0:
                                    #     print(
                                    #         f" new pupil: position-{pupil_pos}, normal_vector-{pupil_normal}, pupil_radius-{pupil_radius}"
                                    #     )

                                    #     image = np.transpose(
                                    #         _input.cpu().numpy(), (1, 2, 0)
                                    #     ).copy()
                                    #     image = visualise_pupil.draw_point(
                                    #         image, eye_model.estimated_eye_center_2D
                                    #     )
                                    #     image = visualise_pupil.draw_normal_vectors_2D(
                                    #         image,
                                    #         utils.projection(pupil_pos, self.focal_len),
                                    #         pupil_normal[0:2],
                                    #         color=(255, 0, 0),
                                    #     )
                                    #     plt.imshow(image)
                                    #     plt.show()

                                except NoIntersection as err:
                                    print(err.message)
                                    row.update(empty_pupil_result)
                                except Exception:
                                    print(err)
                            else:
                                row.update(empty_pupil_result)
                                print("żaden z wektorów nie pokazuje od środka oka")
                        else:
                            row.update(empty_pupil_result)
                            print("No pupil ellipse found")
                    else:
                        row.update(empty_pupil_result)

                writer.writerow(row)
        f.close()
