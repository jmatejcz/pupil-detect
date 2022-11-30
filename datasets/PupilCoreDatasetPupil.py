import torch
from torchvision import transforms
from .utils import get_labels_from_csv, get_frames_from_video
import numpy as np
import cv2
import os


class PupilCoreDataset(torch.utils.data.Dataset):
    """
    Generated with:
    https://github.com/pupil-labs/pupil

    camera_matrix =
        [140.0, 0.0, 96],
        [0.0, 140.0, 96],
        [0.0, 0.0, 1.0],
        ??? # TODO verify
    """

    def __init__(
        self,
        eye0_video_path,
        eye0_labels_path,
        eye1_video_path,
        eye1_labels_path,
        dataset_len,
    ) -> None:
        super().__init__()
        self.eye0_video_path = eye0_video_path
        self.eye0_labels_path = eye0_labels_path
        self.eye1_video_path = eye1_video_path
        self.eye1_labels_path = eye1_labels_path
        self.eye0_labels_df = get_labels_from_csv(self.eye0_labels_path, dataset_len)
        self.eye0_frames = get_frames_from_video(self.eye0_video_path, dataset_len)
        self.eye1_labels_df = get_labels_from_csv(self.eye1_labels_path, dataset_len)
        self.eye1_frames = get_frames_from_video(self.eye1_video_path, dataset_len)
        self.dataset_len = dataset_len
        self.eye0_masks = []
        self.eye1_masks = []
        self.focal_len = 140  # in pixels?

    def load_masks(self, eye0_path, eye1_path):

        for i in range(self.dataset_len):

            mask = cv2.imread(f"{eye0_path}/{i}.png", cv2.IMREAD_GRAYSCALE)
            self.eye0_masks.append(mask)

            mask = cv2.imread(f"{eye1_path}/{i}.png", cv2.IMREAD_GRAYSCALE)
            self.eye1_masks.append(mask)

    def get_pupil_ellipse(self):
        self.eye0_masks = []
        self.eye1_masks = []
        for i, image in enumerate(self.eye0_frames):
            pupil_center = np.array(
                [
                    int(self.eye0_labels_df.at[i, "pupil_center_x_coord"]),
                    int(self.eye0_labels_df.at[i, "pupil_center_y_coord"]),
                ]
            )
            pupil_axes = (
                int(self.eye0_labels_df.at[i, "pupil_axis_1"] // 2),
                int(self.eye0_labels_df.at[i, "pupil_axis_2"] // 2),
            )
            mask = np.zeros(image.shape)
            ellipse = cv2.ellipse(
                img=mask,
                center=pupil_center,
                axes=pupil_axes,
                angle=self.eye0_labels_df.at[i, "elipse_angle"],
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1,
            )
            ellipse = np.asarray(ellipse, dtype=np.float32)
            ellipse = cv2.cvtColor(ellipse, cv2.COLOR_RGB2GRAY)
            ret, b_ellipse = cv2.threshold(ellipse, 1, 255, cv2.THRESH_BINARY)
            b_ellipse = b_ellipse.reshape(b_ellipse.shape + (1,))

            self.eye0_masks.append(b_ellipse)

        for i, image in enumerate(self.eye1_frames):
            pupil_center = np.array(
                [
                    int(self.eye1_labels_df.at[i, "pupil_center_x_coord"]),
                    int(self.eye1_labels_df.at[i, "pupil_center_y_coord"]),
                ]
            )
            pupil_axes = (
                int(self.eye1_labels_df.at[i, "pupil_axis_1"] // 2),
                int(self.eye1_labels_df.at[i, "pupil_axis_2"] // 2),
            )
            mask = np.zeros(image.shape)
            ellipse = cv2.ellipse(
                img=mask,
                center=pupil_center,
                axes=pupil_axes,
                angle=self.eye1_labels_df.at[i, "elipse_angle"],
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1,
            )
            ellipse = np.asarray(ellipse, dtype=np.float32)
            ellipse = cv2.cvtColor(ellipse, cv2.COLOR_RGB2GRAY)
            ret, b_ellipse = cv2.threshold(ellipse, 1, 255, cv2.THRESH_BINARY)
            b_ellipse = b_ellipse.reshape(b_ellipse.shape + (1,))
            self.eye1_masks.append(b_ellipse)

    def save_masks(self, path):
        for i, mask in enumerate(self.eye0_masks):

            cv2.imwrite(f"{path}/eye0/{i}.png", mask)

        for i, mask in enumerate(self.eye1_masks):

            cv2.imwrite(f"{path}/eye1/{i}.png", mask)

    # ================= for pupil segmentation ===================
    def __getitem__(self, idx):
        image = self.eye0_frames[idx]
        pupil_mask = self.eye0_masks[idx]
        opened = self.eye0_labels_df.at[idx, "opened"]
        T = transforms.Compose([transforms.ToTensor()])
        image = T(image)
        pupil_mask = T(pupil_mask)

        return (image, pupil_mask, opened)

    # ================ for pupil core =======================
    # def __getitem__(self, idx):
    #     pupil_coords = np.array(
    #         [
    #             self.eye0_labels_df.at[idx, "pupil_center_x_coord"],
    #             self.eye0_labels_df.at[idx, "pupil_center_y_coord"],
    #         ]
    #     )
    #     image = self.eye0_frames[idx]
    #     T = transforms.Compose([transforms.ToTensor()])
    #     image = T(image)
    #     pupil_coords = torch.as_tensor(pupil_coords, dtype=torch.float32)
    #     return image, pupil_coords

    def __len__(self):
        return len(self.eye0_frames)
