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

warnings.filterwarnings("error")


class GazeTracker:
    def __init__(self, weight_path: str, px_to_mm: int = 30) -> None:
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
        self.inital_eye_center_z = 51 * self.MMTOPX

    def draw_current_pupil_vectors(self, image, eye_model):
        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_model.ellipse_centers[-1],
            eye_model.disc_normals[-1][0][0:2].ravel(),
            color=(255, 0, 0),
        )
        image = visualise_pupil.draw_normal_vectors_2D(
            image,
            eye_model.ellipse_centers[-1],
            eye_model.disc_normals[-1][1][0:2].ravel(),
            color=(0, 255, 0),
        )

    def fit_tracker(self, dataset: PupilCoreDataset):
        dataloader = get_one_dataloader(dataset)
        dataset.get_pupil_masks()
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
                        except NoEllipseFound as err:
                            print(err.message)

                        if ellipse:
                            eye_model.two_circle_unprojection(ellipse)
                            # this part is for visalization
                            # ============================================================
                            # if i % 100 == 0:
                            #     if x == 0:
                            #         base_right_image = visualise_pupil.draw_ellipse(
                            #             base_right_image, ellipse
                            #         )
                            #         base_right_image = self.draw_current_pupil_vectors(
                            #             image=base_right_image, eye_model=eye_model
                            #         )

                            #     else:
                            #         base_left_image = visualise_pupil.draw_ellipse(
                            #             base_left_image, ellipse
                            #         )
                            #         base_left_image = self.draw_current_pupil_vectors(
                            #             image=base_left_image, eye_model=eye_model
                            #         )
                            #     image = np.transpose(
                            #         _input[0].cpu().numpy(), (1, 2, 0)
                            #     ).copy()
                            #     image = visualise_pupil.draw_ellipse(image, ellipse)
                            #     image = self.draw_current_pupil_vectors(
                            #             image=image, eye_model=eye_model
                            #         )
                            #     plt.imshow(image)
                            #     plt.show()
                            # ===========================================================

        for eye in self.eyes_models:
            eye.sphere_centre_estimate()
            eye.sphere_radius_estimate()
            print(f"estimated eye center in 2D -> {eye.estimated_eye_center_2D}")
            print(f"estimated eye center in 3D -> {eye.estimated_eye_center_3D}")
            print(f"estimated eye radius -> {eye.estimated_sphere_radius}")

        base_right_image = visualise_pupil.draw_point(
            base_right_image, self.eyes_models[0].estimated_eye_center_2D
        )
        base_left_image = visualise_pupil.draw_point(
            base_left_image, self.eyes_models[1].estimated_eye_center_2D
        )
        plt.imshow(base_right_image)
        plt.show()
        plt.imshow(base_left_image)
        plt.show()

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
                            # image = np.transpose(_input.cpu().numpy(), (1, 2, 0)).copy()
                            # image = visualise_pupil.draw_point(
                            #     image, eye_model.estimated_eye_center_2D
                            # )
                            # image = self.draw_current_pupil_vectors(
                            #     image=image, eye_model=eye_model
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
