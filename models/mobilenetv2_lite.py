import torch

from torchvision.models.mobilenet import mobilenet_v2

from models.conv import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MobilentV2Lite(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True).features
        self.model1, self.model2 = self._seperate_model()
        self.conv_last = Conv(1280 + 384, 640, 1, 1, 0)
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
        x = torch.ones(2, 3, 416, 416)
        for i, m in enumerate(self.model):
            x = m(x)
            if x.shape[2:] == (13, 13):
                break
        model1 = self.model[:i]
        model2 = self.model[i:]

        return model1, model2

    def forward(self, x):
        x = self.model1(x)
        skip = x.view(x.shape[0], -1, 13, 13)
        x = self.model2(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_last(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = MobilentV2().cuda()
    summary(model, (3, 416, 416))