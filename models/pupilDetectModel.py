from torchvision import models
from torch import nn

# TODO model do wykrywania środka źrenicy, na wyjsciu 2 pola regresji (wspolrzedne)
class PupilDetectModel(nn.Module):
    def __init__(self):
        super(PupilDetectModel, self).__init__()
        self.fc = nn.Linear(in_features=512, out_features=2, dtype=torch.float32)
