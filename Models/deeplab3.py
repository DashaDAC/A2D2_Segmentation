import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

n_classes = 4

class Up_Sample_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Up_Sample_Conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Nearest neighbour for upsampling are two
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Repeat(nn.Module):
    def __init__(self, ch_out):
        super(Repeat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        for i in range(2):
            if i == 0:
                x_rec = self.conv(x)
            x_rec = self.conv(x + x_rec)
        return x_rec

class RR_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(RR_Conv, self).__init__()
        self.Repeat_block = nn.Sequential(Repeat(ch_out), Repeat(ch_out))
        self.Conv = nn.Conv2d(ch_in, ch_out, 1, 1, 0)

    def forward(self, input_img):
        input_img = self.Conv(input_img)
        conv_input_img = self.Repeat_block(input_img)
        return input_img + conv_input_img

# in process
class Atrous_Conv(nn.Module):
    def __init__(self, ch_out):
        super(Atrous_Conv, self).__init__()

        self.pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2) #?
            )

        self.a_conv_1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

        self.a_conv_6 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=6, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

        self.a_conv_12 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=12, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

        self.a_conv_18 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=18, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self, x, ch_out):
        y = self.pooling(x)
        y_1 = self.a_conv_1(x)
        y_6 = self.a_conv_6(x)
        y_12 = self.a_conv_12(x)
        y_18 = self.a_conv_18(x)
        concat = torch.cat([y, y_1, y_6, y_12, y_18],  dim=4) # number of dim =  4?
        end_concat = self.a_conv_1(concat)
        return end_concat



    ############

class DeepLabV3(nn.Module):

    def __init__(self, img_ch=3, output_ch = n_classes):
        super(DeepLabV3, self).__init__()

        return