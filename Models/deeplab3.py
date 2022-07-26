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

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.channel_1 = 64
        self.channel_2 = 2 * self.channel_1
        self.channel_3 = 2 * self.channel_2
        self.channel_4 = 2 * self.channel_3
        self.channel_5 = 2 * self.channel_4

        # For new layer added
        self.channel_6 = 2 * self.channel_5

        self.channels = [self.channel_1, self.channel_2, self.channel_3, self.channel_4, self.channel_5, self.channel_6]

        self.Layer1 = RR_Conv(img_ch, self.channels[0])
        self.Layer2 = RR_Conv(self.channels[0], self.channels[1])
        self.Layer3 = RR_Conv(self.channels[1], self.channels[2])
        self.Layer4 = RR_Conv(self.channels[2], self.channels[3])
        self.Layer5 = RR_Conv(self.channels[3], self.channels[4])
        self.Layer6 = RR_Conv(self.channels[4], self.channels[5])

        self.DeConvLayer6 = Up_Sample_Conv(self.channels[5], self.channels[4])
        self.DeConvLayer5 = Up_Sample_Conv(self.channels[4], self.channels[3])
        self.DeConvLayer4 = Up_Sample_Conv(self.channels[3], self.channels[2])
        self.DeConvLayer3 = Up_Sample_Conv(self.channels[2], self.channels[1])
        self.DeConvLayer2 = Up_Sample_Conv(self.channels[1], self.channels[0])

        self.Up_Layer6 = RR_Conv(self.channels[5], self.channels[4])
        self.Up_Layer5 = RR_Conv(self.channels[4], self.channels[3])
        self.Up_Layer4 = RR_Conv(self.channels[3], self.channels[2])
        self.Up_Layer3 = RR_Conv(self.channels[2], self.channels[1])
        self.Up_Layer2 = RR_Conv(self.channels[1], self.channels[0])

        self.Conv = nn.Conv2d(self.channels[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv1 = self.Layer1(x)
        mp1 = self.MaxPool(conv1)
        conv2 = self.Layer2(mp1)
        mp2 = self.MaxPool(conv2)
        conv3 = self.Layer3(mp2)
        mp3 = self.MaxPool(conv3)
        conv4 = self.Layer4(mp3)
        mp4 = self.MaxPool(conv4)
        conv5 = self.Layer5(mp4)
        mp5 = self.MaxPool(conv5)
        conv6 = self.Layer6(mp5)

        deconv6 = self.DeConvLayer6(conv6)
        deconv6 = torch.cat((conv5, deconv6), dim=1)
        deconv6 = self.Up_Layer6(deconv6)

        deconv5 = self.DeConvLayer5(deconv6)
        deconv5 = torch.cat((conv4, deconv5), dim=1)
        deconv5 = self.Up_Layer5(deconv5)
        deconv4 = self.DeConvLayer4(deconv5)
        deconv4 = torch.cat((conv3, deconv4), dim=1)
        deconv4 = self.Up_Layer4(deconv4)
        deconv3 = self.DeConvLayer3(deconv4)
        deconv3 = torch.cat((conv2, deconv3), dim=1)
        deconv3 = self.Up_Layer3(deconv3)
        deconv2 = self.DeConvLayer2(deconv3)
        deconv2 = torch.cat((conv1, deconv2), dim=1)
        deconv2 = self.Up_Layer2(deconv2)
        deconv1 = self.Conv(deconv2)

        return deconv1