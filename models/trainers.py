import torch
import copy
import numpy as np

from datasets.PupilCoreDatasetPupil import PupilCoreDataset


class PupilSegmentationTrainer:
    def __init__(self, model, dataset: PupilCoreDataset, dataset_len: int) -> None:
        self.model = model
        self.dataloaders = PupilSegmentationTrainer.get_dataloaders(
            dataset, dataset_len, train_split=0.8
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )
        self.weights_path = "models/weights/resnet50.pt"

    @staticmethod
    def get_dataloaders(
        dataset: PupilCoreDataset, dataset_len: int, train_split: float
    ) -> dict:
        train_part = int(dataset_len * train_split)
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

    def get_dice_score(self, pred, mask):
        pred = pred > 0
        dice_score = 2 * (pred * mask).sum() / (pred + mask).sum()
        return dice_score

    def train(self, device, num_epochs: int) -> None:
        self.model = self.model.to(device)
        self.model.train()

        best_model = copy.deepcopy(self.model.state_dict())
        self.best_loss = 1000
        for epoch in range(num_epochs):

            for phase in ["train", "test"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()

                losses = []
                running_loss = 0.0

                for i, (inputs, mask, opened) in enumerate(self.dataloaders[phase]):
                    # print(inputs, mask)
                    inputs = inputs.to(device)
                    mask = mask.to(device)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
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

                    dice_score = self.get_dice_score
                    if i % 100 == 0:
                        print(
                            f"phase: {phase}, batch: {i}/{len(self.dataloaders[phase])}, loss: {np.mean(losses):.4f}, dice score: {dice_score:.4f}"
                        )

                epoch_loss = running_loss / len(self.dataloaders[phase])
                if phase == "train":
                    self.scheduler.step(epoch_loss)

                print(f"phase: {phase}, epoch: {epoch}, epoch_loss: {epoch_loss:.4f}")

                if phase == "test" and epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    best_model = copy.deepcopy(self.model.state_dict())

                    torch.save(best_model, "models/weights/resnet50.pt")


class IfOpenedTrainer:
    def __init__(self, model, dataset: PupilCoreDataset, dataset_len: int) -> None:
        self.model = model
        self.dataloaders = PupilSegmentationTrainer.get_dataloaders(
            dataset, dataset_len, train_split=0.8
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

        self.weights_path = "models/weights/squeezenet1_1.pt"

    @staticmethod
    def get_dataloaders(
        dataset: PupilCoreDataset, dataset_len: int, train_split: float
    ) -> dict:
        train_part = int(dataset_len * train_split)
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

    def train(self, device, num_epochs: int) -> None:
        self.model = self.model.to(device)
        self.model.train()

        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = 1000
        for epoch in range(num_epochs):

            for phase in ["train", "test"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()

                losses = []
                running_loss = 0.0
                running_corrects = 0
                for i, (inputs, mask, opened) in enumerate(self.dataloaders[phase]):
                    # print(inputs, mask)
                    inputs = inputs.to(device)
                    opened = opened.to(device)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    outputs = outputs.to(device)
                    loss = self.criterion(outputs, opened)
                    # print(loss)
                    # print(pred)
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    losses.append(loss.item())
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == opened.data)
                    if i % 100 == 0:
                        print(
                            f"phase: {phase}, batch: {i}/{len(self.dataloaders[phase])}, loss: {np.mean(losses)}"
                        )
                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase])
                print(
                    f"phase: {phase}, epoch{epoch}, epoch_loss: {epoch_loss}, epoch_acc :{epoch_acc}"
                )

                if phase == "test" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(self.model.state_dict())

                torch.save(best_model, "models/weights/squeeznet1_1.pt")
