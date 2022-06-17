import torch.nn as nn
from .basic import ConvBlock, DownsampleBlock, ResidualBlock, SkipConnection, UpsampleBlock

class ResidualHourGlass2(nn.Module):
    #! The same encoder architecture as original implementation of Invertible Grayscale
    def __init__(self, resNum=4, inChannel=3, outChannel=1):
        super(ResidualHourGlass2, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residualHead = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = UpsampleBlock(256, 128)
        self.up1 = UpsampleBlock(128, 64)
        self.residualRear = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, outChannel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f1 = self.residualHead(f1)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.up2(r3) + f2
        r1 = self.up1(r2) + f1
        r1 = self.residualRear(r1)
        y = self.outConv(r1)
        return y


class ResNet(nn.Module):
    #! The same decoder architecture as the original implementation of Invertible Grayscale
    def __init__(self, resNum=8, inChannel=1, outChannel=3):
        super(ResNet, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residual = nn.Sequential(*[ResidualBlock(64) for _ in range(resNum)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        r2 = self.residual(f1)
        y = self.outConv(r2)
        return y


class ColorGenNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=3, is_mpdist=False):
        super(ColorGenNet, self).__init__()
        if is_mpdist:
            BNFunc = nn.SyncBatchNorm
        else:
            BNFunc = nn.BatchNorm2d
        # conv1: 256
        conv1_2 = [nn.Conv2d(inChannel, 64, 3, stride=1, padding=1), nn.ReLU(True),]
        conv1_2 += [nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(True),]
        conv1_2 += [BNFunc(64, affine=True)]
        # conv2: 128
        conv2_2 = [nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True),]
        conv2_2 += [nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(True),]
        conv2_2 += [BNFunc(128, affine=True)]
        # conv3: 64
        conv3_3 = [nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(True),]
        conv3_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv3_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv3_3 += [BNFunc(256, affine=True)]
        # conv4: 32
        conv4_3 = [nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(True),]
        conv4_3 += [nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(True),]
        conv4_3 += [nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(True),]
        conv4_3 += [BNFunc(512, affine=True)]
        # conv5: 32
        conv5_3 = [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv5_3 += [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv5_3 += [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv5_3 += [BNFunc(512, affine=True)]
        # conv6: 32
        conv6_3 = [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv6_3 += [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv6_3 += [nn.Conv2d(512, 512, 3, dilation=2, stride=1, padding=2), nn.ReLU(True),]
        conv6_3 += [BNFunc(512, affine=True)]
        # conv7: 32
        conv7_3 = [nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(True),]
        conv7_3 += [nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(True),]
        conv7_3 += [nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(True),]
        conv7_3 += [BNFunc(512, affine=True)]
        # conv8: 64
        conv8up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 256, 3, stride=1, padding=1),]
        conv3short8 = [nn.Conv2d(256, 256, 3, stride=1, padding=1),]
        conv8_3 = [nn.ReLU(True),]
        conv8_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv8_3 += [nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),]
        conv8_3 += [BNFunc(256, affine=True)]
        # conv9: 128
        conv9up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, 3, stride=1, padding=1),]
        conv2short9 = [nn.Conv2d(128, 128, 3, stride=1, padding=1),]
        conv9_2 = [nn.ReLU(True),]
        conv9_2 += [nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(True),]
        conv9_2 += [BNFunc(128, affine=True)]
        # conv10: 64
        conv10up = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 64, 3, stride=1, padding=1),]
        conv1short10 = [nn.Conv2d(64, 64, 3, stride=1, padding=1),]
        conv10_2 = [nn.ReLU(True),]
        conv10_2 += [nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.LeakyReLU(negative_slope=.2),]
        self.conv1_2 = nn.Sequential(*conv1_2)
        self.conv2_2 = nn.Sequential(*conv2_2)
        self.conv3_3 = nn.Sequential(*conv3_3)
        self.conv4_3 = nn.Sequential(*conv4_3)
        self.conv5_3 = nn.Sequential(*conv5_3)
        self.conv6_3 = nn.Sequential(*conv6_3)
        self.conv7_3 = nn.Sequential(*conv7_3)
        self.conv8up = nn.Sequential(*conv8up)
        self.conv3short8 = nn.Sequential(*conv3short8)
        self.conv8_3 = nn.Sequential(*conv8_3)
        self.conv9up = nn.Sequential(*conv9up)
        self.conv2short9 = nn.Sequential(*conv2short9)
        self.conv9_2 = nn.Sequential(*conv9_2)
        self.conv10up = nn.Sequential(*conv10up)
        self.conv1short10 = nn.Sequential(*conv1short10)
        self.conv10_2 = nn.Sequential(*conv10_2)
        # regression output
        self.model_out = nn.Sequential(*[nn.Conv2d(64, outChannel, kernel_size=1, padding=0, stride=1),])

    def forward(self, input_grays, train_stage=0):
        f1_2 = self.conv1_2(input_grays)
        f2_2 = self.conv2_2(f1_2)
        f3_3 = self.conv3_3(f2_2)
        f4_3 = self.conv4_3(f3_3)
        f5_3 = self.conv5_3(f4_3)
        f6_3 = self.conv6_3(f5_3)
        f7_3 = self.conv7_3(f6_3)
        f8_up = self.conv8up(f7_3) + self.conv3short8(f3_3)
        f8_3 = self.conv8_3(f8_up)
        f9_up = self.conv9up(f8_3) + self.conv2short9(f2_2)
        f9_2 = self.conv9_2(f9_up)
        f10_up = self.conv10up(f9_2) + self.conv1short10(f1_2)
        f10_2 = self.conv10_2(f10_up)
        out_reg = self.model_out(f10_2)
        return out_reg