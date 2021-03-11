import torch
from torch import nn
from temporal_model import TemporalModel


def create_conv_block(in_channels, out_channels, width: int, height: int, kernel_size=3, pool_size=3):
    ret = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(pool_size, pool_size), stride=1),
    )
    # conv
    width = width - (kernel_size - 1)
    height = height - (kernel_size - 1)
    # maxpool
    width = width - (pool_size - 1)
    height = height - (pool_size - 1)
    return ret, width, height


class PoseCNN(TemporalModel):
    def __init__(self, num_classes: int, lr=0.0001, *args, **kwargs):
        super().__init__(lr, num_classes, *args, **kwargs)

    def init_layers(
            self,
            n_keypoints: int,
            n_consecutive_frames: int,
            n_channels=2,
            inplanes=16,
    ):
        self.layer1, width, height = create_conv_block(n_channels, inplanes, n_consecutive_frames, n_keypoints)
        print(width, height)

        self.layer2, width, height = create_conv_block(inplanes, inplanes, width, height)
        print(width, height)

        self.fc1 = nn.Linear(inplanes * width * height, self.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        # print(x.shape)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = torch.softmax(x, -1)
        return x
