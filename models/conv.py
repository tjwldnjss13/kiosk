import torch.nn as nn

activation_dict = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'relu6': nn.ReLU6(True)}


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, use_bn=True, use_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.acti = nn.LeakyReLU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, channel_scale_factor):
        super().__init__()
        mid_channels = in_channels * channel_scale_factor
        self.conv1 = Conv(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = DSConv(mid_channels, mid_channels, 3, stride, 1)
        self.conv3 = Conv(mid_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, use_batch_norm=True):
        super(DSConv, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.dconv = DConv(in_channels, kernel_size, stride, padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.activation = activation_dict['relu6']
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dconv(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1x1(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)

        return x


class DConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.dconv.weight, mode='fan_out')

    def forward(self, x):
        return self.dconv(x)


if __name__ == '__main__':
    from torchsummary import summary
    model = InvertedResidualBlock(3, 16, 2, 6).cuda()
    summary(model, (3, 416, 416))