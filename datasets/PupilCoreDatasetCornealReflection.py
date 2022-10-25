import torch
from torchvision import transforms
from .utils import get_labels_from_csv, get_frames_from_video
import numpy as np


class PupilCoreDatasetCornealReflection(torch.utils.data.Dataset):
    def __init__(
        self, eye0_video_path, eye0_labels_path, eye1_video_path, eye1_labels_path
    ) -> None:
        super().__init__()
        self.eye0_video_path = eye0_video_path
        self.eye0_labels_path = eye0_labels_path
        self.eye1_video_path = eye1_video_path
        self.eye1_labels_path = eye1_labels_path
        self.eye0_labels_df = get_labels_from_csv(self.eye0_labels_path)
        self.eye0_frames = get_frames_from_video(self.eye0_video_path)
        self.eye1_labels_df = get_labels_from_csv(self.eye1_labels_path)
        self.eye1_frames = get_frames_from_video(self.eye1_video_path)

    def __getitem__(self, idx):
        corneal_coords = np.array(
            [
                self.eye0_labels_df.at[idx, "corneal_reflection_x_coord"],
                self.eye0_labels_df.at[idx, "corneal_reflection_y_coord"],
            ]
        )
        image = self.eye0_frames[idx]
        T = transforms.Compose([transforms.ToTensor()])
        image = T(image)
        corneal_coords = torch.as_tensor(corneal_coords, dtype=torch.float32)
        return image, corneal_coords

    def __len__(self):
        return len(self.eye0_frames)
