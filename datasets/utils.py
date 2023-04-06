import cv2
import pandas as pd
import torch


def get_frames_from_video(video_path, data_len):
    if not data_len:
        data_len = 1_000_000
    frames = []
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = (int(width), int(height))
        for i in range(data_len):
            try:
                is_success, frame = cap.read()
            except cv2.error:
                print("err")
                continue
            if not is_success:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(image)
    cap.release()
    return frames


def get_labels_from_csv(csv_path, data_len):
    if not data_len:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, nrows=data_len)
    opened = [1 if x > 0 else 0 for x in df["x"]]
    df.rename(
        columns={
            "x": "pupil_center_x_coord",
            "y": "pupil_center_y_coord",
            "a": "pupil_axis_1",
            "b": "pupil_axis_2",
            "p": "elipse_angle",
        },
        inplace=True,
    )
    df["opened"] = opened
    return df


def get_dataloaders(
    dataset: torch.utils.data.Dataset, dataset_len: int = None, train_split: float = 0.8
) -> dict:
    if not dataset_len:
        dataset_len = len(dataset)
    train_part = int(dataset_len * train_split)
    indices = torch.randperm(len(dataset)).tolist()

    train_set = torch.utils.data.Subset(dataset, indices[:train_part])
    test_set = torch.utils.data.Subset(dataset, indices[train_part:])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    dataloaders = {"train": train_dataloader, "test": test_dataloader}
    return dataloaders


def get_one_dataloader(dataset: torch.utils.data.Dataset, dataset_len: int = None):
    if not dataset_len:
        dataset_len = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader
