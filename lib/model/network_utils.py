import torch.nn as nn

def conv(in_chs, out_chs, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding),
        nn.LeakyReLU(0.1)
    )

class IdentityBlock(nn.Module):
    def __init__(self, channels, filters):
        super(IdentityBlock, self).__init__()
        self.channels = channels
        self.filters = filters
        self.net = nn.Sequential(
            nn.Conv2d(channels, filters[0], 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2])
        )
        self.channel_net = nn.Conv2d(channels, filters[2], 1, 1, 0)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.net(x)
        if self.channels != self.filters[2]:
            y = self.channel_net(x) + y
        else:
            y = x + y
        y = self.relu(y)

        return y


class ConvBlock(nn.Module):
    def __init__(self, channels, filters, stride=2):
        super(ConvBlock, self).__init__()
        self.channels = channels
        self.filters = filters
        self.net = nn.Sequential(
            nn.Conv2d(channels, filters[0], 1, stride=stride),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2])
        )
        self.downsample =  nn.Sequential(
            nn.Conv2d(channels, filters[2], 1, stride=stride),
            nn.BatchNorm2d(filters[2])
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.net(x)
        x = self.downsample(x)
        y = x + y
        y = self.relu(y)

        return y


class DeConvBlock(nn.Module):
    def __init__(self, channels, filters, stride=2):
        super(DeConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels, filters[0], 1),
            nn.Conv2d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2])
        )
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels, filters[2], 1),
            nn.BatchNorm2d(filters[2])
        )
        self.batch_relu = nn.Sequential(
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.net(x)
        x = self.upsample(x)
        x = x + y
        x = self.batch_relu(x)

        return x