from datasets.PupilCoreDatasetPupil import PupilCoreDatasetTraining
import torch
from models.ifOpened import ifOpenedModel
from models.pupilDetectModel import pupilSegmentationModel
from models.trainers import PupilSegmentationTrainer, IfOpenedTrainer

# ======================================================================
# TRAINING OF CNNS =====================================================
DATASET_PATH = "datasets/PupilCoreDataset/"
DATASET_LEN_TO_USE = 3000
dataset = PupilCoreDatasetTraining(
    f"{DATASET_PATH}video5_eye0_video.avi",
    f"{DATASET_PATH}video5_eye0_pupildata.csv",
    f"{DATASET_PATH}video5_eye1_video.avi",
    f"{DATASET_PATH}video5_eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset_1 = PupilCoreDatasetTraining(
    f"{DATASET_PATH}1/eye0_video.avi",
    f"{DATASET_PATH}1/eye0_pupildata.csv",
    f"{DATASET_PATH}1/eye1_video.avi",
    f"{DATASET_PATH}1/eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset_2 = PupilCoreDatasetTraining(
    f"{DATASET_PATH}2/eye0_video.avi",
    f"{DATASET_PATH}2/eye0_pupildata.csv",
    f"{DATASET_PATH}2/eye1_video.avi",
    f"{DATASET_PATH}2/eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset_3 = PupilCoreDatasetTraining(
    f"{DATASET_PATH}3/eye0_video.avi",
    f"{DATASET_PATH}3/eye0_pupildata.csv",
    f"{DATASET_PATH}3/eye1_video.avi",
    f"{DATASET_PATH}3/eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset_4 = PupilCoreDatasetTraining(
    f"{DATASET_PATH}4/eye0_video.avi",
    f"{DATASET_PATH}4/eye0_pupildata.csv",
    f"{DATASET_PATH}4/eye1_video.avi",
    f"{DATASET_PATH}4/eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)
dataset_5 = PupilCoreDatasetTraining(
    f"{DATASET_PATH}5/eye0_video.avi",
    f"{DATASET_PATH}5/eye0_pupildata.csv",
    f"{DATASET_PATH}5/eye1_video.avi",
    f"{DATASET_PATH}5/eye1_pupildata.csv",
    DATASET_LEN_TO_USE,
)

ultimate_dataset = torch.utils.data.ConcatDataset(
    [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ifOpenedModel = ifOpenedModel()
pupilSegmentationModel = pupilSegmentationModel()

if_opened_trainer = IfOpenedTrainer(
    model=ifOpenedModel, dataset=ultimate_dataset, dataset_len=DATASET_LEN_TO_USE * 6
)
pupil_trainer = PupilSegmentationTrainer(
    model=pupilSegmentationModel,
    dataset=ultimate_dataset,
    dataset_len=DATASET_LEN_TO_USE * 6,
    weights_path="models/weights/resnet50.pt",
)

if_opened_trainer.train(device=device, num_epochs=5)
if_opened_trainer.eval_model(device=device)

pupil_trainer.train(device=device, num_epochs=3)
pupil_trainer.eval_model(device=device)
