import torch
import torch.nn as nn

from torchvision.models.mobilenet import mobilenet_v2

from models.conv import *
from models.fortest.attention_modules_modified import CBAM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MobilentV2CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True).features
        self.model1, self.model2, self.model3, self.model4, self.model5 = self._seperate_model()
        self.conv_last = Conv(1280 + 384, 1280, 1, 1, 0)
        self.cbam1 = CBAM(16, 1/16)
        self.cbam2 = CBAM(24, 1/16)
        self.cbam3 = CBAM(32, 1/16)
        self.cbam4 = CBAM(96, 1/16)
        del self.model

    def _init_weights(self):
        for m in self.conv_last.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _seperate_model(self):
        models = []

        _x = x = torch.ones(2, 3, 416, 416)
        prev_size = 0
        idx_begin = 0
        for i, m in enumerate(self.model):
            x = m(x)
            if prev_size and x.shape[-1] < prev_size:
                models.append(self.model[idx_begin:i])
                idx_begin = i
            prev_size = x.shape[-1]
        models.append(self.model[idx_begin:])

        return models

    def forward(self, x):
        x = self.model1(x)
        x = self.cbam1(x)

        x = self.model2(x)
        x = self.cbam2(x)

        x = self.model3(x)
        x = self.cbam3(x)

        x = self.model4(x)
        skip = x = self.cbam4(x)

        x = self.model5(x)

        skip = skip.view(x.shape[0], -1, 7, 7)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_last(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = MobilentV2CBAM().cuda()
    summary(model, (3, 224, 224))
