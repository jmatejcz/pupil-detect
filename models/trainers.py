import torch
import copy
import numpy as np

from datasets.PupilCoreDatasetPupil import PupilCoreDatasetPupil


class PupilSegmentationTrainer:
    def __init__(self, model, dataset: PupilCoreDatasetPupil, dataset_len: int) -> None:
        self.model = model
        self.dataloaders = PupilSegmentationTrainer.get_dataloaders(
            dataset, dataset_len, train_split=0.8
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )

    @staticmethod
    def get_dataloaders(
        dataset: PupilCoreDatasetPupil, dataset_len: int, train_split: float
    ) -> dict:
        train_part = int(dataset_len * 0.8)
        indices = torch.randperm(len(dataset)).tolist()

        train_set = torch.utils.data.Subset(dataset, indices[:train_part])
        test_set = torch.utils.data.Subset(dataset, indices[train_part:])

        train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=8, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=True
        )
        dataloaders = {"train": train_dataloader, "test": test_dataloader}
        return dataloaders

    def train(self, model, device, num_epochs):
        self.model = model.to(device)
        self.model.train()

        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = 1000
        for epoch in range(num_epochs):

            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()

                losses = []
                running_loss = 0.0

                for i, (inputs, mask) in enumerate(self.dataloaders[phase]):
                    # print(inputs, mask)
                    inputs = inputs.to(device)
                    mask = mask.to(device)
                    self.optimizer.zero_grad()

                    outputs = model(inputs)
                    pred = outputs["out"]
                    pred = pred.to(device)
                    loss = self.criterion(pred, mask)
                    # print(loss)
                    # print(pred)
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item()
                    losses.append(loss.item())
                    if i % 100 == 0:
                        print(
                            f"phase: {phase}, batch: {i}/{len(self.dataloaders[phase])}, loss: {np.mean(losses)}"
                        )

                epoch_loss = running_loss / len(self.dataloaders[phase])
                print(f"phase: {phase}, epoch_loss: {epoch_loss}")

                if phase == "test" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model.state_dict())

                torch.save(best_model, "models/weights/resnet50.pt")
