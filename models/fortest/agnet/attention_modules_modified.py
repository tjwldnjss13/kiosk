import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ChannelPoolingBlock(nn.Module):
    def __init__(self, mode):
        super(ChannelPoolingBlock, self).__init__()
        if mode == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif mode == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x_temp = x.permute(0, 2, 3, 1)
        x = torch.zeros(*x_temp.shape[:3], 1)
        for i, x_batch in enumerate(x_temp):
            x[i] = self.pool(x_batch)
        x = x.permute(0, 3, 1, 2)

        return x


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, channels_mid):
        super(ChannelAttentionBlock, self).__init__()
        self.C = channels
        self.M = channels_mid
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.convs = nn.Sequential(
            nn.Conv2d(self.C, self.M, 1, 1, 0),
            nn.Conv2d(self.M, self.C, 1, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        F_avg, F_max = self.avgpool(x), self.maxpool(x)
        F_avg, F_max = self.convs(F_avg), self.convs(F_max)
        x_out = F_avg + F_max
        mask = self.sigmoid(x_out)
        mask = transforms.Resize((x.shape[2:]))(mask)

        return x * mask


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.avgpool = ChannelPoolingBlock('avg')
        self.maxpool = ChannelPoolingBlock('max')
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        F_avg, F_max = self.avgpool(x), self.maxpool(x)
        x_out = torch.cat([F_avg, F_max], dim=1).to(x.device)
        x_out = self.conv(x_out)
        mask = self.sigmoid(x_out)

        return x * mask


class JAM(nn.Module):
    def __init__(self, channels, channel_rescale_value):
        super(JAM, self).__init__()
        self.C = channels
        self.M = int(channels * channel_rescale_value)
        self.channel_attention = ChannelAttentionBlock(self.C, self.M)
        self.spatial_atteition = SpatialAttentionBlock()
        self.conv1x1 = nn.Conv2d(self.C * 2, self.C, 1, 1)

    def forward(self, x):
        F_channel = self.channel_attention(x)
        F_spatial = self.spatial_atteition(x)
        # F_channel = transforms.Resize((self.H, self.W))(F_channel)
        F_channel = transforms.Resize(x.detach().shape[2:])(F_channel)
        F_spatial = torch.cat([F_spatial for _ in range(self.C)], dim=1)

        # x_out = x + (F_channel + F_spatial) * x
        x_out = self.conv1x1(torch.cat([F_channel * x, F_spatial * x], dim=1)) + x

        return x_out


class CBAM(nn.Module):
    def __init__(self, channels, channels_rescale_value):
        super(CBAM, self).__init__()
        # self.H = height
        # self.W = width
        self.C = channels
        self.M = int(channels * channels_rescale_value)
        # self.conv = nn.Conv2d(self.C, self.C, 1, 1, 0)
        self.channel_attention = ChannelAttentionBlock(self.C, self.M)
        self.spatial_atteition = SpatialAttentionBlock()

    def forward(self, x):
        '''
        # F = self.conv(x)
        F = x
        F_channel = self.channel_attention(F)
        F_channel = transforms.Resize(F.shape[2:4])(F_channel)
        F_ = F * F_channel
        F_spatial = self.spatial_atteition(F_)
        F_spatial = F_spatial.permute(0, 2, 3, 1)
        F_ = F_.permute(0, 2, 3, 1)
        F__ = F_ * F_spatial
        F__ = F__.permute(0, 3, 1, 2)
        x = x + F__
        '''
        x = self.channel_attention(x)
        x = self.spatial_atteition(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    jam = JAM(32, 1.5).cuda()
    ca = ChannelAttentionBlock(32, 48).cuda()
    sa = SpatialAttentionBlock().cuda()
    cp = ChannelPoolingBlock('avg').cuda()
    cbam = CBAM(32, 1.5).cuda()

    # summary(cbam, (512, 12, 24))
    # summary(cbam, (512, 24, 48))
    # summary(cbam, (256, 48, 96))
    # summary(cbam, (128, 96, 192))
    # summary(cbam, (64, 192, 384))
    summary(cbam, (32, 384, 768))
































