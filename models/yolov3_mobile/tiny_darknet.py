import torch.nn as nn

from .conv import Conv
from .attention_modules import CBAM


class TinyDarknet(nn.Module):
    def __init__(self, use_batch_norm):
        super().__init__()
        self.layer1 = nn.Sequential(
            Conv(3, 16, 3, 1, 1, use_batch_norm),
            CBAM(16, 1/16),
            nn.MaxPool2d(2, 2),
            Conv(16, 32, 3, 1, 1, use_batch_norm),
            CBAM(32, 1/16),
            nn.MaxPool2d(2, 2),
            Conv(32, 16, 1, 1, 0, use_batch_norm),
            Conv(16, 128, 3, 1, 1, use_batch_norm),
            Conv(128, 16, 1, 1, 0, use_batch_norm),
            Conv(16, 128, 3, 1, 1, use_batch_norm),
            CBAM(128, 1/16),
            nn.MaxPool2d(2, 2),
            Conv(128, 32, 1, 1, 0, use_batch_norm),
            Conv(32, 256, 3, 1, 1, use_batch_norm),
            Conv(256, 32, 1, 1, 0, use_batch_norm),
            Conv(32, 256, 3, 1, 1, use_batch_norm),
            CBAM(256, 1/16)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv(256, 64, 1, 1, 0, use_batch_norm),
            Conv(64, 512, 3, 1, 1, use_batch_norm),
            Conv(512, 64, 1, 1, 0, use_batch_norm),
            Conv(64, 512, 3, 1, 1, use_batch_norm),
            CBAM(512, 1/16)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv(512, 128, 1, 1, 0, use_batch_norm),
            Conv(128, 512, 3, 1, 1, use_batch_norm)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x1, x2, x3


if __name__ == '__main__':
    from torchsummary import summary
    model = TinyDarknet(False).cuda()
    summary(model, (3, 416, 416))

