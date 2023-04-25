import torch
import copy
import numpy as np
from datasets.utils import get_dataloaders
from datasets.PupilCoreDatasetPupil import PupilCoreDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visualization.visualise_pupil import draw_ellipse


class PupilSegmentationTrainer:
    def __init__(
        self,
        model,
        dataset: PupilCoreDataset,
        dataset_len: int = None,
        weights_path: str = None,
    ) -> None:
        if not dataset_len:
            dataset_len = len(PupilCoreDataset)
        self.model = model
        self.dataloaders = get_dataloaders(dataset, dataset_len, train_split=0.8)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, verbose=True
        )
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))

    def get_dice_score(self, pred, mask):
        pred = np.where(pred > 0.4, 1, 0)
        dice_score = 2 * (pred * mask).sum() / (pred + mask).sum()
        return dice_score

    def get_IoU(self, pred, mask):
        pred = np.where(pred > 0.4, 1, 0)
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum() - intersection
        return intersection / union

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

                    if opened[0]:
                        inputs = inputs.to(device)
                        mask = mask.to(device)
                        self.optimizer.zero_grad()

                        outputs = self.model(inputs)
                        pred = outputs["out"]
                        pred = pred.to(device)
                        loss = self.criterion(pred, mask)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                        running_loss += loss.item()
                        losses.append(loss.item())

                        dice_score = self.get_dice_score(
                            pred.detach().cpu().numpy(), mask.detach().cpu().numpy()
                        )
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

    def eval_model(self, device):
        self.model.eval()
        self.model = self.model.to(device)
        dice_scores = []
        iou_scores = []
        with torch.no_grad():
            for i, (inputs, mask, opened) in enumerate(self.dataloaders["test"]):
                if opened[0]:
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)["out"][0]
                    pred = torch.sigmoid(outputs)

                    pred = pred.cpu().numpy()
                    mask = mask.cpu().numpy()
                    outputs = outputs.cpu().numpy()

                    dice_scores.append(self.get_dice_score(outputs, mask))
                    iou_scores.append(self.get_IoU(outputs, mask))

            print(
                f"mean_dice_score: {np.mean(dice_scores)}, mean_iou: {np.mean(iou_scores)}"
            )


class IfOpenedTrainer:
    def __init__(
        self,
        model,
        dataset: PupilCoreDataset,
        dataset_len: int = None,
        weights_path: str = None,
    ) -> None:
        if not dataset_len:
            dataset_len = len(PupilCoreDataset)
        self.model = model
        self.dataloaders = get_dataloaders(dataset, dataset_len, train_split=0.8)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))

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
                    inputs = inputs.to(device)
                    opened = opened.to(device)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    outputs = outputs.to(device)
                    loss = self.criterion(outputs, opened)
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

    def eval_model(self, device):
        """Evalurate model based on diffrent metrics"""
        self.model.eval()
        # confusion matrix
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        self.model = self.model.to(device)
        with torch.no_grad():
            for i, (inputs, mask, opened) in enumerate(self.dataloaders["test"]):
                inputs = inputs.to(device)
                opened = opened.to(device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                outputs = outputs.to(device)

                label = int(opened[0])
                pred = int(preds[0])
                if pred == 1 and label == 1:
                    true_positive += 1

                elif pred == 1 and label == 0:
                    false_positive += 1

                elif pred == 0 and label == 1:
                    false_negative += 1

                elif pred == 0 and label == 0:
                    true_negative += 1
                    true_negative += 1

            acc = (
                (true_positive + true_negative)
                / (true_positive + true_negative + false_negative + false_positive)
                * 100
            )

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * precision * recall / (precision + recall)

            cf_mattrix = [
                [true_positive, false_positive],
                [false_negative, true_negative],
            ]
            ax = sns.heatmap(cf_mattrix, annot=True)
            ax.set_title("Confusion Matrix")
            ax.set_ylabel("Predicted values")
            ax.set_xlabel("Actual values")
            ax.xaxis.set_ticklabels(["1", "0"])
            ax.yaxis.set_ticklabels(["1", "0"])
            plt.show()
            # plt.savefig(f"{path_to_save}/cf")

            ax = sns.heatmap(cf_mattrix / np.sum(cf_mattrix), annot=True, fmt=".2%")
            ax.set_title("Confusion Matrix in percents")
            ax.set_ylabel("Predicted values")
            ax.set_xlabel("Actual values")
            ax.xaxis.set_ticklabels(["1", "0"])
            ax.yaxis.set_ticklabels(["1", "0"])
            plt.show()

            print(
                f"acc: {acc}, precision: {precision}, recall: {recall}, f1_score: {f1_score}"
            )
