import torch
from torchvision import transforms
from .utils import get_labels_from_csv, get_frames_from_video
import numpy as np
import cv2
import pandas as pd


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
        dataset_len=None,
    ) -> None:
        super().__init__()
        self.eye0_video_path = eye0_video_path
        self.eye0_labels_path = eye0_labels_path
        self.eye1_video_path = eye1_video_path
        self.eye1_labels_path = eye1_labels_path
        if dataset_len is not None:
            self.dataset_len = dataset_len
        self.eye0_labels_df = get_labels_from_csv(self.eye0_labels_path, dataset_len)
        self.eye0_frames = get_frames_from_video(self.eye0_video_path, dataset_len)
        self.eye1_labels_df = get_labels_from_csv(self.eye1_labels_path, dataset_len)
        self.eye1_frames = get_frames_from_video(self.eye1_video_path, dataset_len)
        self.eye0_masks = []
        self.eye1_masks = []
        self.focal_len = 140
        self.image_shape = self.eye0_frames[0].shape

    def load_masks(self, eye0_path, eye1_path):

        for i in range(self.dataset_len):
            mask = cv2.imread(f"{eye0_path}/{i}.png", cv2.IMREAD_GRAYSCALE)
            self.eye0_masks.append(mask)

            mask = cv2.imread(f"{eye1_path}/{i}.png", cv2.IMREAD_GRAYSCALE)
            self.eye1_masks.append(mask)

    def get_pupil_masks(self):
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
            _, b_ellipse = cv2.threshold(ellipse, 1, 1, cv2.THRESH_BINARY)
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
            _, b_ellipse = cv2.threshold(ellipse, 1, 255, cv2.THRESH_BINARY)
            b_ellipse = b_ellipse.reshape(b_ellipse.shape + (1,))
            self.eye1_masks.append(b_ellipse)

    def save_masks(self, path):
        for i, mask in enumerate(self.eye0_masks):
            cv2.imwrite(f"{path}/eye0/{i}.png", mask)

        for i, mask in enumerate(self.eye1_masks):
            cv2.imwrite(f"{path}/eye1/{i}.png", mask)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.eye0_frames)


class PupilCoreDatasetTraining(PupilCoreDataset):
    "dataset used for training, __getitem__ returns 1 eye at a time, with its mask"

    def __init__(
        self,
        eye0_video_path,
        eye0_labels_path,
        eye1_video_path,
        eye1_labels_path,
        dataset_len=None,
    ) -> None:
        super().__init__(
            eye0_video_path,
            eye0_labels_path,
            eye1_video_path,
            eye1_labels_path,
            dataset_len,
        )
        self.get_pupil_masks()
        self.eye_frames = self.eye0_frames + self.eye1_frames
        self.eye_masks = self.eye0_masks + self.eye1_masks
        self.eye_labels_df = pd.DataFrame(
            pd.concat(
                [self.eye0_labels_df, self.eye1_labels_df], axis=0, ignore_index=True
            )
        )

    # ================= for pupil segmentation ===================
    def __getitem__(self, idx):
        image = self.eye_frames[idx]
        mask = self.eye_masks[idx]
        opened = self.eye_labels_df.at[idx, "opened"]
        T = transforms.Compose([transforms.ToTensor()])
        image = T(image)
        mask = T(mask)

        return (image, mask, opened)


class PupilCoreDatasetGazeTrack(PupilCoreDataset):
    "dataset used for gaze tracking, __getitem__ returns both eyes parallel"

    def __init__(
        self,
        eye0_video_path,
        eye0_labels_path,
        eye1_video_path,
        eye1_labels_path,
        dataset_len=None,
    ) -> None:
        super().__init__(
            eye0_video_path,
            eye0_labels_path,
            eye1_video_path,
            eye1_labels_path,
            dataset_len,
        )

    # ================= for gaze tracking ===================
    def __getitem__(self, idx):
        try:
            image0 = self.eye0_frames[idx]
            image1 = self.eye1_frames[idx]
        except:
            print(idx)
        opened0 = self.eye0_labels_df.at[idx, "opened"]
        opened1 = self.eye1_labels_df.at[idx, "opened"]

        T = transforms.Compose([transforms.ToTensor()])
        image0 = T(image0)
        image1 = T(image1)

        return ((image0, image1), (opened0, opened1))
