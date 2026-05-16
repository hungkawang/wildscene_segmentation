"""
Reference: The author of the main architecture of the code is:
https://github.com/LilLouis5/, Included:
    1. DoubleConv
    2. Unet
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, features=None):
        super(Unet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.u_sample = nn.ModuleList()
        self.d_sample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.d_sample.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in reversed(features):
            self.u_sample.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=3, stride=1, padding=1))
            self.u_sample.append(DoubleConv(in_channel=feature*2, out_channel=feature))

        self.bottle_neck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []    # to retain features

        for down in self.d_sample:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.u_sample), 2):
            x = self.u_sample[idx](x)

            skip_connection = skip_connections[idx // 2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat = torch.cat((x, skip_connection), dim=1)
            x = self.u_sample[idx + 1](concat)

        return self.final_conv(x)


# def main():
#     x = torch.randn((2, 3, 2016, 1512))
#     model = Unet(in_channel=3, out_channel=19)
#     preds = model(x)
#     print(x.shape)
#     print(preds.shape)
#
#
# if __name__ == '__main__':
#     main()
