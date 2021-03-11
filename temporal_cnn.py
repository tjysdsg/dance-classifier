from resnet import ResNet, resnet101
from torch import nn
from torch.functional import F
from temporal_model import TemporalModel


class TemporalCNN(TemporalModel):
    def __init__(self, num_classes: int, n_consecutive_frames=6, lr=0.001):
        super().__init__(num_classes=num_classes, lr=lr, n_consecutive_frames=n_consecutive_frames)

    def init_layers(self, n_consecutive_frames: int):
        self.model: ResNet = resnet101(pretrained=True, num_classes=self.num_classes, n_channels=n_consecutive_frames)
        self.fc = nn.Linear(1000, self.num_classes)

    def forward(self, x):
        x = self.model.forward(x)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x
